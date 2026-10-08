import csv
import functools
import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

# Every metric takes (values [R, ...], targets [R, D]) and returns {name: float} or a Figure.


def activation_stats(values, targets):
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "absmax": float(np.abs(finite).max()),
        "zeros": float((finite == 0).mean()),
        "nonfinite": float(1 - finite.size / values.size),
    }


def dead_units(values, targets, unit_axis=None):
    """Share of units never > 0. unit_axis=1 for the channels of a conv output [N, C, T]."""
    units = values.reshape(len(values), -1) if unit_axis is None else np.moveaxis(values, unit_axis, -1)
    units = units.reshape(-1, units.shape[-1])
    return {"dead_units": float(1 - (units > 0).any(axis=0).mean())}


def effective_rank(values, targets):
    """Number of dimensions actually used: 1 = collapsed, F = all of them."""
    rows = values.reshape(len(values), -1)
    s = np.linalg.svd(rows - rows.mean(axis=0), compute_uv=False)
    p = s[s > 0] / s.sum()
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum()))}


def linear_probe(values, targets, max_dim=64):
    """Cross-validated R2 of a ridge regression from the values to the targets."""
    rows = values.reshape(len(values), -1)
    reduce = PCA(max_dim) if rows.shape[1] > max_dim else "passthrough"
    model = make_pipeline(StandardScaler(), reduce, Ridge())
    folds = KFold(5, shuffle=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(model, rows, targets, cv=folds, scoring="r2")
    return {"linear_probe_r2": float(scores.mean())}


def target_cka(values, targets):
    """Linear CKA between values and targets, 0 to 1: how aligned the representation is with the task."""
    x = values.reshape(len(values), -1).astype(np.float64)
    x, y = x - x.mean(axis=0), targets - targets.mean(axis=0)
    xx = np.linalg.norm(x.T @ x) if x.shape[1] <= x.shape[0] else np.linalg.norm(x @ x.T)
    return {"target_cka": float(np.linalg.norm(y.T @ x) ** 2 / (xx * np.linalg.norm(y.T @ y) + 1e-12))}


def saturation(values, targets, low=0.0, high=1.0, margin=0.02):
    """Share of values at the bounds of a bounded activation (sigmoid: 0, 1; tanh: -1, 1)."""
    tol = margin * (high - low)
    return {"sat_low": float((values <= low + tol).mean()), "sat_high": float((values >= high - tol).mean())}


def channel_activity(values, targets, channel_axis=1):
    """Conv outputs: dead channels, how evenly channels are used (1 = evenly), mean |corr| between them."""
    flat = np.moveaxis(values, channel_axis, -1).reshape(-1, values.shape[channel_axis]).astype(np.float64)
    energy = (flat**2).mean(axis=0)
    p = energy[energy > 0] / energy.sum()
    live = flat[:, flat.std(axis=0) > 0][:100_000]
    corr = np.corrcoef(live, rowvar=False) if live.shape[1] > 1 else np.eye(1)
    return {
        "dead_channels": float((energy <= 1e-3 * np.median(energy)).mean()),
        "channel_usage": float(np.exp(-(p * np.log(p)).sum()) / len(energy)),
        "channel_redundancy": float(np.abs(corr[~np.eye(len(corr), dtype=bool)]).mean()) if len(corr) > 1 else 0.0,
    }


def temporal_dynamics(values, targets, time_axis=1):
    """Sequences [N, T, F] (time_axis=-1 for conv [N, C, T]): lag-1 autocorrelation and relative step change."""
    v = np.moveaxis(values, time_axis, 1).reshape(len(values), values.shape[time_axis], -1).astype(np.float64)
    a, b = v[:, :-1].reshape(-1, v.shape[2]), v[:, 1:].reshape(-1, v.shape[2])
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    corr = (a * b).sum(axis=0) / np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0) + 1e-12)
    step = np.linalg.norm(np.diff(v, axis=1), axis=2).mean() / (np.linalg.norm(v, axis=2).mean() + 1e-12)
    return {"lag1_autocorr": float(corr.mean()), "step_change": float(step)}


def attention_entropy(values, targets):
    """Attention weights [N, T] (oldest -> newest): entropy (1 = uniform) and lag (0 = newest step)."""
    w = values.reshape(-1, values.shape[-1])
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    T = w.shape[1]
    entropy = -(w * np.log(w + 1e-12)).sum(axis=1).mean() / np.log(T)
    lag = (T - 1 - (w * np.arange(T)).sum(axis=1)).mean() / (T - 1)
    return {"attention_entropy": float(entropy), "attention_lag": float(lag)}


def _scatter(points, targets, title):
    col = targets.var(axis=0).argmax()
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot()
    sc = ax.scatter(points[:, 0], points[:, 1], c=targets[:, col], s=5, cmap="viridis")
    fig.colorbar(sc, label=f"target {col}")
    ax.set_title(title)
    return fig


def pca(values, targets):
    rows = values.reshape(len(values), -1)
    return _scatter(PCA(2).fit_transform(rows), targets, "PCA")


def tsne(values, targets):
    rows = values.reshape(len(values), -1)
    return _scatter(TSNE(2, init="pca", random_state=0).fit_transform(rows), targets, "t-SNE")


def probe(name, *metrics, every=1):
    """Record the output of a model method and compute `metrics` on it every `every` epochs.

    Metrics are names of the functions above or any function (values, targets) -> dict | Figure,
    e.g. functools.partial(saturation, low=-1). Decorators can be stacked (different `every`).
    """
    metrics = [(globals()[m] if isinstance(m, str) else m, every) for m in metrics]

    def decorator(method):
        if hasattr(method, "_probe_metrics"):
            method._probe_metrics.extend(metrics)
            return method

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            out = method(self, *args, **kwargs)
            store = getattr(self, "_probe_store", None)
            if store is not None:
                key = f"{self._probe_path}.{name}" if self._probe_path else name
                tensor = out[0] if isinstance(out, tuple) else out
                entry = store.setdefault(key, {"values": [], "metrics": wrapper._probe_metrics})
                entry["values"].append(tensor.detach().float().cpu())
            return out

        wrapper._probe_metrics = list(metrics)
        return wrapper

    return decorator


def _metric_name(fn):
    return getattr(fn, "__name__", None) or fn.func.__name__


class ProbeReporter:
    """Computes the probes of a model on a fixed random subset of `dataset`.

    Writes out_dir/probes/scalars.csv (epoch, probe, metric, value), one plot of the
    scalars per probe, and the figures as out_dir/probes/<probe>/<metric>/<epoch>.png.
    """

    def __init__(self, dataset, out_dir, n_samples=2000, seed=0):
        idx = np.random.default_rng(seed).choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        self.loader = DataLoader(Subset(dataset, np.sort(idx)), batch_size=512)
        self.out_dir = Path(out_dir) / "probes"
        self.rows = []

    @torch.no_grad()
    def collect(self, model):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        modules = list(model.named_modules())
        store = {}
        for path, m in modules:
            m._probe_store, m._probe_path = store, path
        targets = []
        try:
            for x, y in self.loader:
                model(x.to(device))
                targets.append(y)
        finally:
            for _, m in modules:
                m._probe_store = None
            model.train(was_training)
        targets = torch.cat(targets).numpy().astype(np.float64)
        for entry in store.values():
            values = torch.cat(entry["values"]).numpy()
            entry["values"] = values
            entry["targets"] = np.repeat(targets, len(values) // len(targets), axis=0)
        return store

    def report(self, model, epoch, tag=None):
        """Metrics due at this epoch; with a tag (e.g. "best") all of them, saved under that tag."""
        tag = tag or f"epoch{epoch:03d}"
        for probe_name, entry in self.collect(model).items():
            for fn, every in entry["metrics"]:
                if tag.startswith("epoch") and epoch % every:
                    continue
                result = fn(entry["values"], entry["targets"])
                if isinstance(result, Figure):
                    path = self.out_dir / probe_name / _metric_name(fn) / f"{tag}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    result.savefig(path, dpi=120, bbox_inches="tight")
                else:
                    self.rows += [(epoch, tag, probe_name, k, v) for k, v in result.items()]
        self.save()

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "scalars.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "tag", "probe", "metric", "value"])
            writer.writerows(self.rows)
        for probe_name in {r[2] for r in self.rows}:
            rows = [r for r in self.rows if r[2] == probe_name and r[1].startswith("epoch")]
            keys = sorted({r[3] for r in rows})
            if not keys:
                continue
            cols = min(4, len(keys))
            fig = Figure(figsize=(4 * cols, 3 * -(-len(keys) // cols)), layout="constrained")
            for i, key in enumerate(keys):
                ax = fig.add_subplot(-(-len(keys) // cols), cols, i + 1)
                points = sorted((r[0], r[4]) for r in rows if r[3] == key)
                ax.plot(*zip(*points, strict=True), marker="o", ms=3)
                ax.set_title(key)
                ax.grid(alpha=0.3)
            fig.suptitle(probe_name)
            (self.out_dir / probe_name).mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / probe_name / "scalars.png", dpi=100)

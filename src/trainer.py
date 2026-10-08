import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.metrics import regression_metrics
from src.probes import ProbeReporter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, loss_fn=None, device="cpu"):
    loss_fn = loss_fn or nn.MSELoss()
    model.eval()
    preds, targets, loss = [], [], 0.0
    for x, y in loader:
        out = model(x.to(device)).cpu()
        loss += loss_fn(out, y).item() * len(y)
        preds.append(out)
        targets.append(y)
    y_true, y_pred = torch.cat(targets).numpy(), torch.cat(preds).numpy()
    return {"loss": loss / len(y_true), **regression_metrics(y_true, y_pred)}


def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip=None):
    model.train()
    totals = {"loss": 0.0}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        for name, child in model.named_children():
            grads = [p.grad.norm() ** 2 for p in child.parameters() if p.grad is not None]
            if grads:
                totals[f"grad_norm.{name}"] = (
                    totals.get(f"grad_norm.{name}", 0.0) + torch.stack(grads).sum().sqrt().item()
                )
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        totals["loss"] += loss.item()
    return {k: v / len(loader) for k, v in totals.items()}


def train(
    model,
    train_loader,
    val_loader,
    out_dir,
    test_loader=None,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    loss_fn=None,
    optimizer=None,
    monitor="r2",
    device="auto",
):
    """Train, keep the best model on val[monitor] (higher is better), probe it and test it.

    Writes to out_dir: best_model.pt, history.csv, metrics.json and probes/.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = loss_fn or nn.MSELoss()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = model.to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    reporter = ProbeReporter(val_loader.dataset, out_dir)

    history, best = [], None
    for epoch in range(1, epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        val_m = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, **{f"train/{k}": v for k, v in train_m.items()}, **_scalars(val_m, "val/")})
        _write_csv(history, out_dir / "history.csv")
        print(f"[{epoch:03d}] train loss {train_m['loss']:.4f} | val loss {val_m['loss']:.4f} r2 {val_m['r2']:.4f}")

        if best is None or val_m[monitor] > best[monitor]:
            best = {"epoch": epoch, **val_m}
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        reporter.report(model, epoch)

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    reporter.report(model, best["epoch"], tag="best")
    results = {"val": _scalars(best)}
    if test_loader is not None:
        results["test"] = _scalars(evaluate(model, test_loader, loss_fn, device))
        print(f"test r2 {results['test']['r2']:.4f} | jitter_ratio {results['test']['jitter_ratio']:.2f}")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def _scalars(metrics, prefix=""):
    return {prefix + k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)

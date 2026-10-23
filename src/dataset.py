import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_recordings(raw_dir, meta_keys):
    """Yield every recording of raw_dir as a dict (see README for the .npz format)."""
    for path in sorted(Path(raw_dir).rglob("*.npz")):
        with np.load(path) as f:
            rec = {k: f[k] for k in f.files}
        rec["meta"] = [int(rec[f"meta_{k}"]) for k in meta_keys]
        yield rec


def select(meta, meta_keys, where):
    """Mask of the rows whose metadata matches where = {key: [values]}."""
    mask = np.ones(len(meta), dtype=bool)
    for key, values in (where or {}).items():
        mask &= np.isin(meta[:, meta_keys.index(key)], values)
    return mask


def make_windows(emg, targets, window, stride, skip=0):
    """[C, T], [D, T] -> windows [N, C, window] and the target at the last sample of each [N, D]."""
    starts = np.arange(skip, emg.shape[1] - window + 1, stride)
    X = np.stack([emg[:, s : s + window] for s in starts])
    y = targets[:, starts + window - 1].T
    return X, y.astype(np.float32)


def build_dataset(params, preprocess):
    cfg, pre = params["dataset"], params["preprocessing"]
    meta_keys = cfg["meta_keys"]
    X, y, meta = [], [], []
    for rec in load_recordings(cfg["raw_dir"], meta_keys):
        if float(rec["fs"]) != pre["fs"]:
            raise ValueError(f"recording at {rec['fs']} Hz, preprocessing expects {pre['fs']} Hz")
        emg = preprocess(rec["emg"], **pre)
        step = round(rec["emg"].shape[1] / emg.shape[1])
        targets = rec["targets"][:, ::step][:, : emg.shape[1]]
        skip = int(cfg.get("skip_start", 0) * pre["fs"] / step)  # filter start-up transient
        Xr, yr = make_windows(emg, targets, cfg["window"], cfg["stride"], skip)
        X.append(Xr)
        y.append(yr)
        meta.append(np.repeat([rec["meta"]], len(Xr), axis=0))
    if not X:
        raise FileNotFoundError(f"no .npz recordings in {cfg['raw_dir']}")
    X, y, meta = np.concatenate(X), np.concatenate(y), np.concatenate(meta).astype(np.int32)

    is_test = select(meta, meta_keys, cfg["test"])
    norm_per = cfg.get("normalize_per", "subject" if "subject" in meta_keys else "global")
    if norm_per == "subject" and "subject" in meta_keys:
        subj_idx = meta_keys.index("subject")
        unique_subjects = np.unique(meta[:, subj_idx])
        for s in unique_subjects:
            s_mask = meta[:, subj_idx] == s
            m = X[s_mask].mean(axis=(0, 2), keepdims=True)
            s_dev = X[s_mask].std(axis=(0, 2), keepdims=True) + 1e-6
            X[s_mask] = (X[s_mask] - m) / s_dev
        subj_list = list(unique_subjects)
        print(f"Standardized EMG independently for {len(unique_subjects)} participants (subjects: {subj_list})")
        mean = X.mean(axis=(0, 2), keepdims=True)[0]
        std = X.std(axis=(0, 2), keepdims=True)[0] + 1e-6
    else:
        mean = X[~is_test].mean(axis=(0, 2), keepdims=True)[0]
        std = X[~is_test].std(axis=(0, 2), keepdims=True)[0] + 1e-6
        X = (X - mean) / std

    out = Path(cfg["out_dir"])
    for split, mask in (("train", ~is_test), ("test", is_test)):
        (out / split).mkdir(parents=True, exist_ok=True)
        np.save(out / split / "X.npy", X[mask].astype(np.float32))
        np.save(out / split / "y.npy", y[mask])
        np.save(out / split / "meta.npy", meta[mask])
    np.savez(out / "stats.npz", mean=mean, std=std)
    info = {
        "channel_names": [str(c) for c in rec["channel_names"]],
        "target_names": [str(t) for t in rec["target_names"]],
        "meta_keys": meta_keys,
        "fs": pre["fs"] / step,
        "window": cfg["window"],
        "stride": cfg["stride"],
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(f"{int((~is_test).sum())} train / {int(is_test.sum())} test windows -> {out}")


class WindowDataset(Dataset):
    def __init__(self, data_dir, split="train", where=None, exclude=False):
        data_dir = Path(data_dir)
        self.info = json.loads((data_dir / "info.json").read_text())
        self.X = np.load(data_dir / split / "X.npy", mmap_mode="r")
        self.y = np.load(data_dir / split / "y.npy")
        mask = select(np.load(data_dir / split / "meta.npy"), self.info["meta_keys"], where)
        self.indices = np.flatnonzero(~mask if exclude else mask)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        i = self.indices[i]
        return torch.from_numpy(np.array(self.X[i])), torch.from_numpy(self.y[i])


def make_loaders(params):
    """Train (minus the val selection), val and test loaders. Val/test keep time order."""
    data_dir, cfg = params["dataset"]["out_dir"], params["train"]
    train = WindowDataset(data_dir, "train", cfg["val"], exclude=True)
    val = WindowDataset(data_dir, "train", cfg["val"])
    test = WindowDataset(data_dir, "test")
    bs = cfg["batch_size"]
    return (
        DataLoader(train, bs, shuffle=True, drop_last=True),
        DataLoader(val, bs),
        DataLoader(test, bs) if len(test) else None,
    )

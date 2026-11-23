"""WaveFormer sEMG dataset discovery, loading, windowing, and augmentations."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.preprocess import Preprocess, preprocess_waveformer_window

CLASS_NAMES_6: tuple[str, ...] = (
    "Relax",
    "Open",
    "Close",
    "Wave-In",
    "Wave-Out",
    "Pinch",
)
CLASS_NAMES_3: tuple[str, ...] = ("Relax", "Open", "Close")

REACTEMG_DOWNLOAD_URL: str = (
    "https://www.dropbox.com/scl/fi/isj4450alriqjfstkna2s/data.zip?rlkey=n5sf910lopskewzyae0vgn6j7&st=vt89hfpj&dl=1"
)

_EPN_3CLASS_KEYWORDS: tuple[str, ...] = ("relax", "open", "fist", "close")


def _extract_emg_and_gt_from_csv(path: Path, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Load raw EMG channels and ground-truth labels from a CSV recording."""
    df = pd.read_csv(path)
    cols_lower = {str(col).strip().lower(): col for col in df.columns}

    emg_cols: list[object] = []
    for prefix in ("emg_", "emg", "ch_", "ch", "channel_"):
        candidate = [cols_lower[f"{prefix}{i}"] for i in range(channels) if f"{prefix}{i}" in cols_lower]
        if len(candidate) == channels:
            emg_cols = candidate
            break

    if not emg_cols:
        excluded = {"gt", "label", "action", "target", "timestamp", "time", "index", "unnamed: 0"}
        numeric_cols = [
            col
            for col in df.columns
            if str(col).strip().lower() not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_cols) < channels:
            raise ValueError(f"Expected at least {channels} EMG columns in {path}, found {list(df.columns)}")
        emg_cols = numeric_cols[:channels]

    raw_emg = df[emg_cols].to_numpy(dtype=np.float32)

    gt_col = None
    for candidate_name in ("gt", "label", "action", "target"):
        if candidate_name in cols_lower:
            gt_col = cols_lower[candidate_name]
            break

    if gt_col is None:
        actions = np.full(len(df), -1, dtype=np.int64)
    else:
        actions = pd.to_numeric(df[gt_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

    return raw_emg, actions


def _extract_emg_and_gt_from_npy(path: Path, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Load raw EMG channels and labels from an NPY array ``[gt, emg0..emg7]``."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    if arr.shape[1] == channels + 1:
        actions = arr[:, 0].astype(np.int64)
        raw_emg = arr[:, 1 : 1 + channels].astype(np.float32)
    elif arr.shape[1] == channels:
        actions = np.full(arr.shape[0], -1, dtype=np.int64)
        raw_emg = arr[:, :channels].astype(np.float32)
    else:
        raise ValueError(
            f"Expected {channels + 1} ([gt, emg...]) or {channels} columns in {path}, got shape {arr.shape}"
        )
    return raw_emg, actions


def load_raw_recording(
    path: str | Path,
    channels: int = 8,
    num_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw unscaled EMG ``(T, C)`` float32 and valid labels ``(T,)`` int64."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw_emg, actions = _extract_emg_and_gt_from_csv(file_path, channels=channels)
    elif suffix == ".npy":
        raw_emg, actions = _extract_emg_and_gt_from_npy(file_path, channels=channels)
    else:
        raise ValueError(f"Unsupported recording format {suffix!r} for {file_path}")

    valid_mask = (actions >= 0) & (actions < int(num_classes))
    actions = np.where(valid_mask, actions, -1).astype(np.int64, copy=False)
    return raw_emg, actions


def load_reactemg_file(
    path: str | Path,
    preprocess: Preprocess | None = None,
    channels: int = 8,
    num_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and condition a single ``.csv`` or ``.npy`` sEMG recording."""
    raw_emg, actions = load_raw_recording(path, channels=channels, num_classes=num_classes)
    prep = preprocess if preprocess is not None else Preprocess(channels=channels, mode="waveformer")
    emg_tc = prep.preprocess_sequence(raw_emg)
    return emg_tc, actions


def _is_unlabeled_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix().lower()
    except ValueError:
        rel = path.name.lower()
    return "unlabel" in rel


def _is_epn_path(rel_posix: str) -> bool:
    return "emg-epn-612" in rel_posix or "epn" in rel_posix


def _filter_by_dataset_selection(
    labeled_files: list[Path],
    root: Path,
    dataset_selection: str,
    num_classes: int,
    epn_subset_percentage: float,
    rng: np.random.Generator,
) -> list[Path]:
    selection = dataset_selection.lower()
    rel_map = {p: p.relative_to(root).as_posix().lower() for p in labeled_files}

    epn_files = [p for p in labeled_files if _is_epn_path(rel_map[p]) and "testingjson_user" not in rel_map[p]]
    roam_files = [p for p in labeled_files if "roam" in rel_map[p] and not _is_epn_path(rel_map[p])]
    roam_set = set(roam_files)
    other_public = [p for p in labeled_files if not _is_epn_path(rel_map[p]) and p not in roam_set]

    if int(num_classes) == 3 and epn_files:
        if selection == "pub_with_epn":
            open_fist = [p for p in epn_files if "open" in p.name.lower() or "fist" in p.name.lower()]
            if open_fist:
                epn_files = open_fist
        else:
            filtered_epn = [p for p in epn_files if any(k in p.name.lower() for k in _EPN_3CLASS_KEYWORDS)]
            if filtered_epn:
                epn_files = filtered_epn

    if epn_subset_percentage < 1.0 and epn_files:
        epn_pool = sorted(epn_files)
        perm_epn = rng.permutation(len(epn_pool))
        keep_n = max(1, int(round(len(epn_pool) * float(epn_subset_percentage))))
        epn_files = [epn_pool[int(i)] for i in perm_epn[:keep_n]]

    if selection == "epn_only":
        selected = epn_files if epn_files else labeled_files
    elif selection == "roam_only":
        selected = roam_files if roam_files else labeled_files
    elif selection == "pub_with_epn":
        selected = epn_files + roam_files + other_public
        if not selected:
            selected = labeled_files
    elif selection == "custom_folder":
        selected = labeled_files
    else:
        raise ValueError(
            f"Unsupported dataset_selection {dataset_selection!r}; "
            "expected 'pub_with_epn', 'epn_only', 'roam_only', or 'custom_folder'"
        )

    return sorted(dict.fromkeys(selected))


def _fast_scan_epn_subjects(
    epn_root: Path,
    epn_subset_percentage: float,
    val_split_ratio: float,
    rng: np.random.Generator,
) -> dict[str, list[Path]] | None:
    """Fast subject-disjoint split across ``EMG-EPN-612/trainingJSON_user*`` directories."""
    user_dirs = sorted(
        entry.path
        for entry in os.scandir(epn_root)
        if entry.is_dir() and entry.name.lower().startswith("trainingjson_user")
    )
    if len(user_dirs) < 2:
        return None

    perm_u = rng.permutation(len(user_dirs))
    keep_u = max(2, int(round(len(user_dirs) * min(1.0, max(0.01, float(epn_subset_percentage))))))
    keep_u = min(len(user_dirs), keep_u)
    selected_dirs = [Path(user_dirs[int(i)]) for i in perm_u[:keep_u]]

    if val_split_ratio > 0.0:
        val_u = max(1, min(len(selected_dirs) - 1, int(round(len(selected_dirs) * float(val_split_ratio)))))
    else:
        val_u = 0

    val_dirs = selected_dirs[:val_u]
    train_dirs = selected_dirs[val_u:]

    def _collect_files(dirs: Sequence[Path]) -> list[Path]:
        collected: list[Path] = []
        for d in dirs:
            for entry in sorted(os.scandir(d), key=lambda e: e.name):
                if entry.is_file() and entry.name.lower().endswith((".npy", ".csv")):
                    collected.append(Path(entry.path))
        return collected

    return {
        "train": _collect_files(train_dirs),
        "val": _collect_files(val_dirs),
        "unlabeled": [],
    }


def discover_reactemg_files(
    data_root: str | Path,
    dataset_selection: str = "epn_only",
    num_classes: int = 6,
    epn_subset_percentage: float = 1.0,
    val_split_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Discover and split sEMG recording paths into train, val, and unlabeled pools.

    For ``epn_only`` over multi-subject ``EMG-EPN-612/trainingJSON_user*`` folders,
    performs fast **subject-disjoint** discovery (unseen subjects held out in ``val``)
    so validation measures true cross-subject zero-shot generalization.
    """
    root = Path(data_root)
    if not root.exists():
        return {"train": [], "val": [], "unlabeled": []}

    selection_lc = dataset_selection.lower()
    epn_only_root = root / "EMG-EPN-612"
    rng = np.random.default_rng(int(seed))

    if selection_lc == "epn_only" and int(num_classes) == 6 and epn_only_root.is_dir():
        subject_split = _fast_scan_epn_subjects(
            epn_root=epn_only_root,
            epn_subset_percentage=epn_subset_percentage,
            val_split_ratio=val_split_ratio,
            rng=rng,
        )
        if subject_split is not None and subject_split["train"]:
            return subject_split

    walk_root = epn_only_root if (selection_lc == "epn_only" and epn_only_root.is_dir()) else root
    all_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(walk_root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".") and d != "__pycache__" and not d.lower().startswith("testingjson_user")
        )
        for fname in sorted(filenames):
            if fname.lower().endswith((".csv", ".npy")):
                all_files.append(Path(dirpath) / fname)

    unlabeled_files: list[Path] = []
    labeled_files: list[Path] = []
    for p in all_files:
        if _is_unlabeled_path(p, root):
            unlabeled_files.append(p)
        else:
            labeled_files.append(p)

    selected = _filter_by_dataset_selection(
        labeled_files=labeled_files,
        root=root,
        dataset_selection=dataset_selection,
        num_classes=num_classes,
        epn_subset_percentage=epn_subset_percentage,
        rng=rng,
    )

    if not selected:
        return {"train": [], "val": [], "unlabeled": sorted(unlabeled_files)}

    explicit_val: list[Path] = []
    explicit_train: list[Path] = []
    for p in selected:
        rel_parts = {part.lower() for part in p.relative_to(root).parts[:-1]}
        if "val" in rel_parts or "validation" in rel_parts or "test" in rel_parts:
            explicit_val.append(p)
        else:
            explicit_train.append(p)

    if explicit_val:
        perm_tr = rng.permutation(len(explicit_train))
        perm_va = rng.permutation(len(explicit_val))
        train_files = [explicit_train[int(i)] for i in perm_tr]
        val_files = [explicit_val[int(i)] for i in perm_va]
    elif len(selected) == 1 or val_split_ratio <= 0.0:
        perm = rng.permutation(len(selected))
        train_files = [selected[int(i)] for i in perm]
        val_files = []
    else:
        perm = rng.permutation(len(selected))
        shuffled = [selected[int(i)] for i in perm]
        val_count = max(1, int(round(len(shuffled) * float(val_split_ratio))))
        val_count = min(val_count, len(shuffled) - 1)
        val_files = shuffled[:val_count]
        train_files = shuffled[val_count:]

    return {
        "train": train_files,
        "val": val_files,
        "unlabeled": sorted(unlabeled_files),
    }


def _extract_waveformer_windows_from_recording(
    raw_emg: np.ndarray,
    gt: np.ndarray,
    window_size: int,
    channels: int = 8,
    num_classes: int = 6,
    scale_divisor: float = 128.0,
    hand_choice: str = "right",
    remove_dc: bool = True,
    stride: int = 40,
    include_pre_rest: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract balanced sustained, onset-edge, and rest windows of shape ``(C, W)``.

    Per-window DC removal ensures exact parity with real-time streaming inference.
    """
    t_len = raw_emg.shape[0]
    if t_len == 0:
        return [], []

    if t_len < window_size:
        reps = (window_size // t_len) + 1
        raw_emg = np.tile(raw_emg, (reps, 1))[:window_size]
        gt = np.tile(gt, reps)[:window_size]
        t_len = window_size

    def _cond_win(raw_slice: np.ndarray) -> np.ndarray:
        emg_tc = preprocess_waveformer_window(
            raw_slice,
            scale_divisor=scale_divisor,
            hand_choice=hand_choice,
            remove_dc=remove_dc,
            channels=channels,
            return_nchw=False,
        )
        return emg_tc.T  # (C, W)

    windows: list[np.ndarray] = []
    labels: list[int] = []

    valid_gt = gt[gt >= 0]
    if valid_gt.size == 0:
        return [], []

    unique_classes = np.unique(valid_gt)
    # Multi-gesture continuous sequence (e.g., ROAM / synthetic): slide across file
    if len(unique_classes[unique_classes > 0]) > 1:
        max_start = t_len - window_size
        for s in range(0, max_start + 1, max(1, int(stride))):
            e = s + window_size
            tail_gt = gt[s + window_size // 2 : e]
            tail_valid = tail_gt[tail_gt >= 0]
            if tail_valid.size == 0:
                continue
            vals, counts = np.unique(tail_valid, return_counts=True)
            lbl = int(vals[int(np.argmax(counts))])
            if 0 <= lbl < num_classes:
                windows.append(_cond_win(raw_emg[s:e]))
                labels.append(lbl)
        return windows, labels

    # Single trial EPN-612 style: either pure noGesture (max==0) or 1 active gesture c in 1..5
    active_cls = int(np.max(valid_gt))
    if active_cls <= 0:
        # Pure Relax trial: take 3 diverse 1.0s windows matching 3 active windows per active trial
        candidate_starts = sorted({0, max(0, (t_len - window_size) // 2), max(0, t_len - window_size)})
        if t_len >= window_size + 100 and len(candidate_starts) == 3:
            candidate_starts = [50, max(0, (t_len - window_size) // 2), max(0, t_len - window_size - 50)]
        for s in candidate_starts:
            windows.append(_cond_win(raw_emg[s : s + window_size]))
            labels.append(0)
        return windows, labels

    idx = np.where(gt == active_cls)[0]
    g_start, g_end = int(idx[0]), int(idx[-1]) + 1
    act_len = g_end - g_start

    # 1. Sustained 100%-active contraction window (critical for continuous online holds!)
    if act_len >= window_size:
        s_sustained = min(max(0, g_start + (act_len - window_size) // 2), t_len - window_size)
        act_seg = raw_emg[s_sustained : s_sustained + window_size]
        windows.append(_cond_win(act_seg))
        labels.append(active_cls)
        if active_cls == 1:
            rolled_seg = np.roll(act_seg, shift=max(1, len(act_seg) // 2), axis=0)
            windows.append(_cond_win(rolled_seg))
            labels.append(active_cls)
    else:
        seg = raw_emg[g_start:g_end]
        reps = (window_size // len(seg)) + 1
        sustained_raw = np.tile(seg, (reps, 1))[:window_size]
        windows.append(_cond_win(sustained_raw))
        labels.append(active_cls)
        if active_cls == 1:
            rolled_seg = np.roll(seg, shift=max(1, len(seg) // 2), axis=0)
            rolled_raw = np.tile(rolled_seg, (reps, 1))[:window_size]
            windows.append(_cond_win(rolled_raw))
            labels.append(active_cls)

    # 2. Natural centered gesture window from recording
    center = (g_start + g_end) // 2
    s_nat = min(max(0, center - window_size // 2), t_len - window_size)
    windows.append(_cond_win(raw_emg[s_nat : s_nat + window_size]))
    labels.append(active_cls)

    # 3. Onset-edge window: trailing half..3/4 inside active gesture (fast live reaction)
    onset_tail = max(min(act_len, max(80, int(0.65 * window_size))), min(act_len, window_size))
    e_onset = min(t_len, max(window_size, g_start + onset_tail))
    s_onset = e_onset - window_size
    if s_onset != s_nat and g_start < e_onset:
        windows.append(_cond_win(raw_emg[s_onset:e_onset]))
        labels.append(active_cls)
    else:
        # Take shifted second sustained window if onset collides
        s_alt = min(max(0, g_end - window_size), t_len - window_size)
        windows.append(_cond_win(raw_emg[s_alt : s_alt + window_size]))
        labels.append(active_cls)

    # 4. Pre-onset or post-offset pure rest window if strictly rest throughout
    if include_pre_rest:
        if g_start >= window_size + 15:
            s_rest = max(0, g_start - window_size - 15)
            if np.all(gt[s_rest : s_rest + window_size] == 0):
                windows.append(_cond_win(raw_emg[s_rest : s_rest + window_size]))
                labels.append(0)
        elif (t_len - g_end) >= window_size + 30:
            s_rest = g_end + 30
            if s_rest + window_size <= t_len and np.all(gt[s_rest : s_rest + window_size] == 0):
                windows.append(_cond_win(raw_emg[s_rest : s_rest + window_size]))
                labels.append(0)

    # 5. Offset-to-relax cooldown window: previous gesture on left, >=90 samples (450ms) rest on right
    e_release = g_end + 90
    if e_release <= t_len and e_release >= window_size:
        if np.all(gt[g_end:e_release] == 0):
            s_release = e_release - window_size
            windows.append(_cond_win(raw_emg[s_release:e_release]))
            labels.append(0)

    return windows, labels


class WaveFormerDataset(Dataset):
    """RAM-cached WaveFormer 6-class dataset returning ``(emg_1ct, label)`` pairs.

    Applies WaveFormer time-series augmentations during training:
    - Log-uniform amplitude gain scaling ``[0.55, 1.80]`` (gain robustness across users)
    - Small channel-wise gain jitter ``[0.85, 1.15]``
    - Additive Gaussian sensor noise ``N(0, jitter_sigma)``
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        window_size: int = 200,
        channels: int = 8,
        num_classes: int = 6,
        scale_divisor: float = 128.0,
        hand_choice: str = "right",
        remove_dc: bool = True,
        stride: int = 40,
        train: bool = True,
        rescaling_range: tuple[float, float] = (0.55, 1.80),
        jitter_sigma: float = 0.004,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.scale_divisor = float(scale_divisor)
        self.hand_choice = str(hand_choice)
        self.remove_dc = bool(remove_dc)
        self.stride = max(1, int(stride))
        self.train = bool(train)
        self.rescaling_range = (float(rescaling_range[0]), float(rescaling_range[1]))
        self.jitter_sigma = float(jitter_sigma)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.x_data, self.y_data = self._load_or_build_cache(cache_dir)

    def _cache_file(self, cache_dir: str | Path) -> Path:
        h = hashlib.sha256()
        sig = (
            f"v2|w={self.window_size}|c={self.channels}|k={self.num_classes}"
            f"|dc={self.remove_dc}|hc={self.hand_choice}|tr={self.train}|n={len(self.file_paths)}"
        )
        h.update(sig.encode())
        for p in self.file_paths[:5] + self.file_paths[-5:]:
            h.update(str(p).encode())
        split_tag = "train" if self.train else "val"
        return Path(cache_dir) / f"waveformer_{split_tag}_{h.hexdigest()[:12]}.pt"

    def _load_or_build_cache(self, cache_dir: str | Path | None) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._cache_file(cache_dir) if cache_dir is not None and len(self.file_paths) > 100 else None
        if cache_path is not None and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            return payload["x"], payload["y"]

        win_list: list[np.ndarray] = []
        lbl_list: list[int] = []

        def _process_one(idx_path: tuple[int, Path]) -> tuple[list[np.ndarray], list[int]]:
            idx_i, path = idx_path
            try:
                raw_emg, gt = load_raw_recording(path, channels=self.channels, num_classes=self.num_classes)
            except Exception:
                return [], []
            return _extract_waveformer_windows_from_recording(
                raw_emg=raw_emg,
                gt=gt,
                window_size=self.window_size,
                channels=self.channels,
                num_classes=self.num_classes,
                scale_divisor=self.scale_divisor,
                hand_choice=self.hand_choice,
                remove_dc=self.remove_dc,
                stride=self.stride,
                include_pre_rest=not self.train or (idx_i % 2 == 0),
            )

        indexed_paths = list(enumerate(self.file_paths))
        if len(indexed_paths) > 50:
            with ThreadPoolExecutor(max_workers=16) as pool:
                for wins, lbls in tqdm(
                    pool.map(_process_one, indexed_paths, chunksize=32),
                    total=len(indexed_paths),
                    desc=f"Extracting WaveFormer {'train' if self.train else 'val'} windows",
                ):
                    win_list.extend(wins)
                    lbl_list.extend(lbls)
        else:
            for item in indexed_paths:
                wins, lbls = _process_one(item)
                win_list.extend(wins)
                lbl_list.extend(lbls)

        if not win_list:
            x_empty = torch.zeros((0, self.channels, self.window_size), dtype=torch.float16)
            y_empty = torch.zeros((0,), dtype=torch.long)
            return x_empty, y_empty

        x_arr = np.stack(win_list, axis=0)
        y_arr = np.asarray(lbl_list, dtype=np.int64)
        # Keep storage compact as float16 in RAM/disk (max precision loss < 1e-4 on [-1,1])
        x_tensor = torch.from_numpy(x_arr).to(dtype=torch.float16)
        y_tensor = torch.from_numpy(y_arr).to(dtype=torch.long)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"x": x_tensor, "y": y_tensor}, cache_path)

        return x_tensor, y_tensor

    def class_counts(self) -> list[int]:
        counts = torch.bincount(self.y_data, minlength=self.num_classes)
        return [int(c) for c in counts.tolist()]

    def __len__(self) -> int:
        return int(self.y_data.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # (C, W) float32 -> (1, C, W)
        x = self.x_data[idx].to(dtype=torch.float32)
        y = self.y_data[idx]

        if self.train:
            lo, hi = self.rescaling_range
            global_gain = float(np.exp(self._rng.uniform(np.log(lo), np.log(hi))))
            ch_gain = torch.from_numpy(self._rng.uniform(0.85, 1.15, size=(self.channels, 1)).astype(np.float32))
            x = x * (global_gain * ch_gain)
            if self.jitter_sigma > 0.0:
                x = x + torch.randn_like(x) * self.jitter_sigma
            # Random small temporal circular shift for shift invariance inside window
            if self._rng.random() < 0.5:
                shift = int(self._rng.integers(-15, 16))
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)
            if self.remove_dc:
                x = x - x.mean(dim=-1, keepdim=True)

        return x.unsqueeze(0), y


EMGDataset = WaveFormerDataset

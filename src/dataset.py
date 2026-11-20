"""ReactEMG dataset discovery, loading, windowing, and dynamic masking."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from src.masking import masking_a2a
from src.preprocess import Preprocess

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
    """Load raw EMG channels and ground-truth labels from a ReactEMG CSV file."""
    df = pd.read_csv(path)
    cols_lower = {str(col).strip().lower(): col for col in df.columns}

    emg_cols: list[object] = []
    for prefix in ("emg_", "emg", "ch_", "ch", "channel_"):
        candidate = [cols_lower[f"{prefix}{i}"] for i in range(channels) if f"{prefix}{i}" in cols_lower]
        if len(candidate) == channels:
            emg_cols = candidate
            break

    if not emg_cols:
        # Fallback: select first `channels` numeric columns excluding known label/time columns.
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
    """Load raw EMG channels and labels from a ReactEMG NPY array ``[gt, emg0..emg7]``."""
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


def load_reactemg_file(
    path: str | Path,
    preprocess: Preprocess | None = None,
    channels: int = 8,
    num_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single ReactEMG ``.csv`` or ``.npy`` recording.

    Args:
        path: Path to ``.csv`` or ``.npy`` recording file.
        preprocess: Optional :class:`~src.preprocess.Preprocess` instance.
        channels: Expected EMG channel count (default 8).
        num_classes: Gesture class count (3 or 6).

    Returns:
        Tuple ``(emg_tc, actions)`` with ``emg_tc`` of shape ``(T, C)`` float32
        in ``[0, 1]`` and ``actions`` of shape ``(T,)`` int64 (``-1`` if unlabeled).
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw_emg, actions = _extract_emg_and_gt_from_csv(file_path, channels=channels)
    elif suffix == ".npy":
        raw_emg, actions = _extract_emg_and_gt_from_npy(file_path, channels=channels)
    else:
        raise ValueError(f"Unsupported recording format {suffix!r} for {file_path}")

    prep = preprocess if preprocess is not None else Preprocess(channels=channels)
    emg_tc = prep.preprocess_sequence(raw_emg)

    valid_mask = (actions >= 0) & (actions < int(num_classes))
    actions = np.where(valid_mask, actions, -1).astype(np.int64, copy=False)
    return emg_tc, actions


def _is_unlabeled_path(path: Path, root: Path) -> bool:
    """Return True if path corresponds to an unlabeled recording."""
    try:
        rel = path.relative_to(root).as_posix().lower()
    except ValueError:
        rel = path.name.lower()
    return "unlabel" in rel


def _filter_by_dataset_selection(
    labeled_files: list[Path],
    root: Path,
    dataset_selection: str,
    num_classes: int,
    epn_subset_percentage: float,
    rng: np.random.Generator,
) -> list[Path]:
    """Apply dataset selection and 3-class vs 6-class EPN filtering rules."""
    selection = dataset_selection.lower()
    rel_map = {p: p.relative_to(root).as_posix().lower() for p in labeled_files}

    epn_files = [
        p
        for p in labeled_files
        if ("emg-epn-612" in rel_map[p] or "epn" in rel_map[p]) and "testingjson_user" not in rel_map[p]
    ]
    roam_files = [p for p in labeled_files if "roam" in rel_map[p]]
    other_public = [p for p in labeled_files if p not in set(epn_files) and p not in set(roam_files)]

    # Apply EPN 3-class filtering only when num_classes == 3; 6-class keeps all gestures.
    if int(num_classes) == 3 and epn_files:
        filtered_epn: list[Path] = []
        for p in epn_files:
            name_lower = p.name.lower()
            if any(k in name_lower for k in _EPN_3CLASS_KEYWORDS):
                filtered_epn.append(p)
        if filtered_epn:
            epn_files = filtered_epn

    if epn_subset_percentage < 1.0 and epn_files:
        keep_n = max(1, int(round(len(epn_files) * float(epn_subset_percentage))))
        if keep_n < len(epn_files):
            epn_sorted = sorted(epn_files)
            chosen_idx = np.sort(rng.choice(len(epn_sorted), size=keep_n, replace=False))
            epn_files = [epn_sorted[int(i)] for i in chosen_idx]

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


def discover_reactemg_files(
    data_root: str | Path,
    dataset_selection: str = "pub_with_epn",
    num_classes: int = 6,
    epn_subset_percentage: float = 1.0,
    val_split_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Discover and split ReactEMG recording paths into train, val, and unlabeled pools.

    Supports ``pub_with_epn``, ``epn_only``, ``roam_only``, and ``custom_folder`` modes.
    Retains all 6 classes when ``num_classes == 6``, excludes ``"unlabel"`` files from
    the labeled splits, and places explicit ``val`` / ``validation`` directory files into
    ``val`` if present or uses a deterministic ``val_split_ratio`` split otherwise.
    """
    root = Path(data_root)
    if not root.exists():
        return {"train": [], "val": [], "unlabeled": []}

    all_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".npy"})

    unlabeled_files: list[Path] = []
    labeled_files: list[Path] = []
    for p in all_files:
        if _is_unlabeled_path(p, root):
            unlabeled_files.append(p)
        else:
            labeled_files.append(p)

    rng = np.random.default_rng(int(seed))
    selected = _filter_by_dataset_selection(
        labeled_files=labeled_files,
        root=root,
        dataset_selection=dataset_selection,
        num_classes=num_classes,
        epn_subset_percentage=epn_subset_percentage,
        rng=rng,
    )

    if not selected:
        return {"train": [], "val": [], "unlabeled": unlabeled_files}

    explicit_val: list[Path] = []
    explicit_train: list[Path] = []
    for p in selected:
        rel_parts = {part.lower() for part in p.relative_to(root).parts[:-1]}
        if "val" in rel_parts or "validation" in rel_parts:
            explicit_val.append(p)
        elif "test" in rel_parts:
            explicit_val.append(p)
        else:
            explicit_train.append(p)

    if explicit_val:
        train_files = explicit_train
        val_files = explicit_val
    elif len(selected) == 1 or val_split_ratio <= 0.0:
        train_files = selected
        val_files = []
    else:
        perm = rng.permutation(len(selected))
        val_count = max(1, int(round(len(selected) * float(val_split_ratio))))
        val_count = min(val_count, len(selected) - 1)
        val_idx_set = {int(i) for i in perm[:val_count]}
        val_files = [selected[i] for i in range(len(selected)) if i in val_idx_set]
        train_files = [selected[i] for i in range(len(selected)) if i not in val_idx_set]

    return {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "unlabeled": sorted(unlabeled_files),
    }


def _find_first_transition(actions: np.ndarray) -> int:
    """Return index of first label change inside an action window, or -1 if constant."""
    if len(actions) < 2:
        return -1
    diffs = np.where(actions[1:] != actions[:-1])[0]
    if len(diffs) == 0:
        return -1
    return int(diffs[0] + 1)


class ReactEMGDataset(Dataset):
    """PyTorch Dataset returning dynamically masked ReactEMG multimodal 8-tuples.

    Each item returns:
        ``(emg_window, action_window, masked_emg, masked_actions,
        mask_positions_emg, mask_positions_actions, task_idx, transition_index)``
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        window_size: int = 600,
        stride: int = 30,
        channels: int = 8,
        num_classes: int = 6,
        task_selection: Iterable[int] = (0, 1, 2),
        preprocess: Preprocess | None = None,
        mask_alignment: str = "aligned",
        lambda_poisson: float = 7.0,
        sampling_probability_poisson: float = 0.99,
        poisson_mask_range: tuple[float, float] = (0.3, 0.75),
        end_mask_range: tuple[float, float] = (0.3, 0.6),
        transition_buffer: int = 30,
        noise: float = 0.0,
        is_unlabeled: bool = False,
        deterministic: bool = False,
        seed: int = 42,
    ) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.window_size = int(window_size)
        self.stride = max(1, int(stride))
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.mask_token_id = self.num_classes
        self.task_selection = tuple(int(t) for t in task_selection)
        self.preprocess = preprocess if preprocess is not None else Preprocess(channels=self.channels)
        self.mask_alignment = str(mask_alignment)
        self.lambda_poisson = float(lambda_poisson)
        self.sampling_probability_poisson = float(sampling_probability_poisson)
        self.poisson_mask_range = (float(poisson_mask_range[0]), float(poisson_mask_range[1]))
        self.end_mask_range = (float(end_mask_range[0]), float(end_mask_range[1]))
        self.transition_buffer = int(transition_buffer)
        self.noise = float(noise)
        self.is_unlabeled = bool(is_unlabeled)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.recordings_emg: list[np.ndarray] = []
        self.recordings_actions: list[np.ndarray] = []
        self.samples: list[tuple[int, int, int, int]] = []  # (rec_idx, start, task_idx, transition_idx)
        self.labeled_indices: list[int] = []
        self.unlabeled_indices: list[int] = []

        self._load_and_index()

    def _load_and_index(self) -> None:
        for path in tqdm(
            self.file_paths,
            desc="Loading ReactEMG recordings",
            disable=len(self.file_paths) <= 20,
        ):
            emg_tc, actions = load_reactemg_file(
                path,
                preprocess=self.preprocess,
                channels=self.channels,
                num_classes=self.num_classes,
            )
            if len(emg_tc) < self.window_size:
                continue

            rec_idx = len(self.recordings_emg)
            self.recordings_emg.append(emg_tc)
            self.recordings_actions.append(actions)

            tasks_for_file = (3,) if self.is_unlabeled or np.all(actions < 0) else self.task_selection
            max_start = len(emg_tc) - self.window_size
            for start in range(0, max_start + 1, self.stride):
                act_slice = actions[start : start + self.window_size]
                # Skip labeled windows with unknown action labels (-1) unless running unlabeled task
                has_unlabeled_tokens = bool(np.any(act_slice < 0))
                if not self.is_unlabeled and has_unlabeled_tokens:
                    if 3 not in self.task_selection:
                        continue
                    effective_tasks: Iterable[int] = (3,)
                else:
                    effective_tasks = tasks_for_file

                trans_idx = -1 if has_unlabeled_tokens else _find_first_transition(act_slice)
                for task_idx in effective_tasks:
                    sample_pos = len(self.samples)
                    self.samples.append((rec_idx, start, int(task_idx), trans_idx))
                    if int(task_idx) == 3:
                        self.unlabeled_indices.append(sample_pos)
                    else:
                        self.labeled_indices.append(sample_pos)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rec_idx, start, task_idx, trans_idx = self.samples[idx]
        end = start + self.window_size

        emg_win = self.recordings_emg[rec_idx][start:end].copy()
        act_win = self.recordings_actions[rec_idx][start:end].copy()

        rng = np.random.default_rng(self.seed + idx) if self.deterministic else self._rng

        if self.noise > 0.0 and not self.deterministic:
            emg_win = np.clip(
                emg_win + rng.normal(0.0, self.noise, size=emg_win.shape).astype(np.float32),
                0.0,
                None,
            ).astype(np.float32)

        if task_idx == 3:
            act_win = np.full(self.window_size, self.mask_token_id, dtype=np.int64)

        p_emg = float(rng.uniform(self.poisson_mask_range[0], self.poisson_mask_range[1]))
        if task_idx == 0:
            if self.deterministic:
                # Validation measures full-window all-<MASK> inference performance directly.
                act_mask_type = "end"
                p_act = 1.0
            else:
                # Stage-2 curriculum: predominantly 100% action masking so inference-time all-<MASK> works accurately.
                r_stage = float(rng.random())
                if r_stage < 0.80:
                    act_mask_type = "end"
                    p_act = 1.0
                elif r_stage < 0.90 and trans_idx >= 0:
                    act_mask_type = "targeted"
                    p_act = 0.0
                else:
                    act_mask_type = "poisson"
                    p_act = float(rng.uniform(self.poisson_mask_range[0], self.poisson_mask_range[1]))
        elif rng.random() < self.sampling_probability_poisson:
            act_mask_type = "poisson"
            p_act = float(rng.uniform(self.poisson_mask_range[0], self.poisson_mask_range[1]))
        elif trans_idx >= 0 and rng.random() < 0.5:
            act_mask_type = "targeted"
            p_act = 0.0
        else:
            act_mask_type = "end"
            p_act = float(rng.uniform(self.end_mask_range[0], self.end_mask_range[1]))

        emg_channels = list(range(self.channels))

        if task_idx == 0:
            # Predict masked action tokens from clean EMG
            masked_emg = emg_win.copy()
            mask_pos_emg = np.zeros_like(emg_win, dtype=bool)
            masked_actions, mask_pos_actions = masking_a2a(
                act_win,
                mask_percentage=p_act,
                mask_type=act_mask_type,
                mask_channel_selection=[0],
                lambda_poisson=self.lambda_poisson,
                mask_token=self.mask_token_id,
                transition_index=trans_idx,
                transition_buffer=self.transition_buffer,
                rng=rng,
            )
        elif task_idx == 1:
            # Reconstruct masked EMG from clean actions
            masked_emg, mask_pos_emg = masking_a2a(
                emg_win,
                mask_percentage=p_emg,
                mask_type="poisson",
                mask_channel_selection=emg_channels,
                mask_alignment=self.mask_alignment,
                lambda_poisson=self.lambda_poisson,
                mask_token=0.0,
                rng=rng,
            )
            masked_actions = act_win.copy()
            mask_pos_actions = np.zeros_like(act_win, dtype=bool)
        elif task_idx == 2:
            # Joint masked EMG + masked action reconstruction
            masked_emg, mask_pos_emg = masking_a2a(
                emg_win,
                mask_percentage=p_emg,
                mask_type="poisson",
                mask_channel_selection=emg_channels,
                mask_alignment=self.mask_alignment,
                lambda_poisson=self.lambda_poisson,
                mask_token=0.0,
                rng=rng,
            )
            masked_actions, mask_pos_actions = masking_a2a(
                act_win,
                mask_percentage=p_act,
                mask_type=act_mask_type,
                mask_channel_selection=[0],
                lambda_poisson=self.lambda_poisson,
                mask_token=self.mask_token_id,
                transition_index=trans_idx,
                transition_buffer=self.transition_buffer,
                rng=rng,
            )
        elif task_idx == 3:
            # Self-supervised EMG reconstruction with all actions masked out
            masked_emg, mask_pos_emg = masking_a2a(
                emg_win,
                mask_percentage=p_emg,
                mask_type="poisson",
                mask_channel_selection=emg_channels,
                mask_alignment=self.mask_alignment,
                lambda_poisson=self.lambda_poisson,
                mask_token=0.0,
                rng=rng,
            )
            masked_actions = act_win.copy()
            mask_pos_actions = np.zeros_like(act_win, dtype=bool)
        else:
            raise ValueError(f"Unsupported task_idx {task_idx}; expected 0, 1, 2, or 3")

        return (
            torch.from_numpy(emg_win.astype(np.float32, copy=False)),
            torch.from_numpy(act_win.astype(np.int64, copy=False)),
            torch.from_numpy(masked_emg.astype(np.float32, copy=False)),
            torch.from_numpy(masked_actions.astype(np.int64, copy=False)),
            torch.from_numpy(mask_pos_emg.astype(bool, copy=False)),
            torch.from_numpy(mask_pos_actions.astype(bool, copy=False)),
            torch.tensor(task_idx, dtype=torch.long),
            torch.tensor(trans_idx, dtype=torch.long),
        )


class LabeledUnlabeledSampler(Sampler[int]):
    """Sampler mixing labeled (Tasks 0-2) and unlabeled (Task 3) indices across epochs."""

    def __init__(
        self,
        dataset: ReactEMGDataset,
        unlabeled_weight: float = 0.25,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.labeled = np.asarray(dataset.labeled_indices, dtype=np.int64)
        self.unlabeled = np.asarray(dataset.unlabeled_indices, dtype=np.int64)
        self.unlabeled_weight = float(unlabeled_weight)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if len(self.labeled) > 0 and len(self.unlabeled) > 0:
            self._unlabeled_count = max(
                1, min(len(self.unlabeled), int(round(len(self.labeled) * self.unlabeled_weight)))
            )
        elif len(self.labeled) == 0:
            self._unlabeled_count = len(self.unlabeled)
        else:
            self._unlabeled_count = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.labeled) + self._unlabeled_count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        parts: list[np.ndarray] = []
        if len(self.labeled) > 0:
            lab = self.labeled.copy()
            if self.shuffle:
                rng.shuffle(lab)
            parts.append(lab)
        if self._unlabeled_count > 0:
            unlab = self.unlabeled.copy()
            if self.shuffle:
                rng.shuffle(unlab)
            parts.append(unlab[: self._unlabeled_count])

        if not parts:
            return iter([])
        combined = np.concatenate(parts)
        if self.shuffle:
            rng.shuffle(combined)
        return iter(combined.tolist())

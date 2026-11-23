"""Unit tests for WaveFormer preprocessing, dataset discovery, windowing, and legacy compat."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import (
    CLASS_NAMES_6,
    LabeledUnlabeledSampler,
    ReactEMGDataset,
    WaveFormerDataset,
    discover_reactemg_files,
    load_reactemg_file,
)
from src.preprocess import LEFT_HAND_REMAP, Preprocess, WaveFormerPreprocess, preprocess_waveformer_window


def test_waveformer_preprocess_signed_bipolar_and_dc_removal() -> None:
    raw = np.zeros((40, 8), dtype=np.float32)
    # Oscillatory bipolar AC waveform + DC offset on channel 0
    t = np.arange(40, dtype=np.float32)
    raw[:, 0] = 64.0 * np.sin(2.0 * np.pi * t / 10.0) + 12.0
    raw[:, 2] = 32.0 * np.cos(2.0 * np.pi * t / 10.0)

    out_tc = preprocess_waveformer_window(raw, scale_divisor=128.0, hand_choice="right", remove_dc=True)
    assert out_tc.shape == (40, 8)
    assert out_tc.dtype == np.float32
    # Preserves negative and positive bipolar values and removes DC
    assert out_tc[:, 0].min() < -0.35 and out_tc[:, 0].max() > 0.35
    assert abs(float(out_tc[:, 0].mean())) < 1e-5

    prep_left = WaveFormerPreprocess(hand_choice="left", remove_dc=True)
    out_left = prep_left.preprocess_sequence(raw)
    assert LEFT_HAND_REMAP == (6, 5, 4, 3, 2, 1, 0, 7)
    assert np.allclose(out_left[:, 6], out_tc[:, 0], atol=1e-5)
    assert np.allclose(out_left[:, 4], out_tc[:, 2], atol=1e-5)

    # Return NCHW (1, C, T) when requested (preserving channel-first orientation)
    nchw = preprocess_waveformer_window(raw.T, return_nchw=True)
    assert nchw.shape == (1, 8, 40)


def test_legacy_reactemg_preprocess_preserved() -> None:
    raw = np.zeros((25, 8), dtype=np.float32)
    raw[:, 0] = -64.0
    raw[5, 1] = 120.0
    prep_rect = Preprocess(hand_choice="right", mode="reactemg", median_filter_size=3, medfilt_order="before_rec")
    out_r = prep_rect.preprocess_sequence(raw)
    assert np.isclose(out_r[10, 0], 64.0 / 128.0, atol=1e-5)
    assert np.isclose(out_r[5, 1], 0.0, atol=1e-5)


def test_csv_npy_loading_and_subject_disjoint_epn_discovery(tmp_path: Path) -> None:
    assert len(CLASS_NAMES_6) == 6
    epn_root = tmp_path / "EMG-EPN-612"
    for uid in ("trainingJSON_user1", "trainingJSON_user2", "trainingJSON_user3"):
        udir = epn_root / uid
        udir.mkdir(parents=True)
        for cls_id, name in enumerate(("noGesture", "open", "fist", "waveIn", "waveOut", "pinch")):
            arr = np.zeros((250, 9), dtype=np.float32)
            if cls_id > 0:
                arr[80:220, 0] = cls_id
            arr[:, 1:] = 24.0 * np.sin(np.linspace(0, 8 * np.pi, 250))[:, None]
            np.save(udir / f"sample_{name}.npy", arr)

    splits = discover_reactemg_files(
        tmp_path,
        dataset_selection="epn_only",
        num_classes=6,
        epn_subset_percentage=1.0,
        val_split_ratio=0.34,
        seed=42,
    )
    train_users = {p.parent.name for p in splits["train"]}
    val_users = {p.parent.name for p in splits["val"]}
    assert train_users and val_users
    assert train_users.isdisjoint(val_users)

    emg_tc, gt = load_reactemg_file(splits["train"][0], num_classes=6)
    assert emg_tc.shape == (250, 8) and gt.shape == (250,)


def test_waveformer_dataset_window_extraction_and_augmentation(tmp_path: Path) -> None:
    files: list[Path] = []
    rng = np.random.default_rng(11)
    for cls_id, name in enumerate(("noGesture", "open", "fist", "waveIn", "waveOut", "pinch")):
        arr = np.zeros((400, 9), dtype=np.float32)
        arr[:, 1:] = rng.normal(0.0, 2.0, size=(400, 8))
        if cls_id > 0:
            arr[120:320, 0] = cls_id
            arr[120:320, 1:] += rng.normal(0.0, 20.0, size=(200, 8))
        p = tmp_path / f"rec_{name}.npy"
        np.save(p, arr)
        files.append(p)

    ds_train = WaveFormerDataset(files, window_size=100, channels=8, num_classes=6, train=True, seed=7)
    assert len(ds_train) >= 12
    x0, y0 = ds_train[0]
    assert x0.shape == (1, 8, 100)
    assert x0.dtype == torch.float32
    assert 0 <= int(y0.item()) < 6
    assert all(c > 0 for c in ds_train.class_counts())

    ds_val = WaveFormerDataset(files, window_size=100, channels=8, num_classes=6, train=False, seed=7)
    x_val, _ = ds_val[0]
    assert abs(float(x_val.mean().item())) < 1e-3


def test_legacy_reactemg_dataset_compat(tmp_path: Path) -> None:
    p = tmp_path / "seq.csv"
    gt = np.concatenate([np.zeros(60, dtype=np.int64), np.full(60, 5, dtype=np.int64)])
    df = pd.DataFrame({"gt": gt, **{f"emg_{c}": np.full(120, 64, dtype=np.int64) for c in range(8)}})
    df.to_csv(p, index=False)

    ds = ReactEMGDataset([p], window_size=120, stride=10, num_classes=6, task_selection=(0, 3), deterministic=True)
    assert len(ds) == 2
    sampler = LabeledUnlabeledSampler(ds, seed=5)
    assert len(list(iter(sampler))) == 2

"""Unit tests for ReactEMG preprocessing, masking, file discovery, and dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import (
    CLASS_NAMES_6,
    LabeledUnlabeledSampler,
    ReactEMGDataset,
    discover_reactemg_files,
    load_reactemg_file,
)
from src.masking import masking_a2a
from src.preprocess import LEFT_HAND_REMAP, Preprocess


def test_preprocess_medfilt_rectify_scale_and_lefthand_remap() -> None:
    raw = np.zeros((25, 8), dtype=np.float32)
    # Single spike on ch 1 at t=5 should be suppressed by median filter before rectification
    raw[:, 0] = -64.0
    raw[5, 1] = 120.0
    raw[:, 2] = 32.0

    prep_right = Preprocess(hand_choice="right", median_filter_size=3, medfilt_order="before_rec")
    out_r = prep_right.preprocess_sequence(raw)
    assert out_r.shape == (25, 8)
    assert out_r.dtype == np.float32
    assert np.isclose(out_r[10, 0], 64.0 / 128.0, atol=1e-5)
    assert np.isclose(out_r[5, 1], 0.0, atol=1e-5)

    prep_left = Preprocess(hand_choice="left", median_filter_size=3, medfilt_order="after_rec")
    out_l = prep_left.preprocess_sequence(raw)
    # Under LEFT_HAND_REMAP=(6,5,4,3,2,1,0,7), out[:, 6]=raw[:, 0] (64/128) and out[:, 4]=raw[:, 2] (32/128)
    assert LEFT_HAND_REMAP == (6, 5, 4, 3, 2, 1, 0, 7)
    assert np.isclose(out_l[10, 6], 64.0 / 128.0, atol=1e-5)
    assert np.isclose(out_l[10, 4], 32.0 / 128.0, atol=1e-5)


def test_masking_a2a_modes() -> None:
    rng = np.random.default_rng(7)
    actions = np.repeat(np.arange(4), 30).astype(np.int64)  # length 120, transitions at 30, 60, 90

    masked_p, pos_p = masking_a2a(actions, 0.5, "poisson", [0], mask_token=6, rng=rng)
    assert masked_p.shape == (120,)
    assert pos_p.shape == (120,)
    assert int(pos_p.sum()) == 60
    assert np.all(masked_p[pos_p] == 6)

    masked_end, pos_end = masking_a2a(actions, 0.25, "end", [0], mask_token=6, rng=rng)
    assert np.all(pos_end[-30:]) and not np.any(pos_end[:-30])
    assert np.all(masked_end[-30:] == 6)

    masked_tgt, pos_tgt = masking_a2a(
        actions, 0.0, "targeted", [0], mask_token=6, transition_index=60, transition_buffer=10, rng=rng
    )
    assert int(pos_tgt.sum()) == 20
    assert np.all(pos_tgt[50:70])
    assert np.all(masked_tgt[50:70] == 6)


def test_csv_npy_loading_and_6class_discovery(tmp_path: Path) -> None:
    assert len(CLASS_NAMES_6) == 6
    epn_dir = tmp_path / "EMG-EPN-612" / "s1"
    roam_dir = tmp_path / "ROAM_EMG" / "s1"
    epn_dir.mkdir(parents=True)
    roam_dir.mkdir(parents=True)

    # Create CSVs covering all 6 classes (including wavein, waveout, pinch) and an unlabel file
    for name, cls_id in [("wavein.csv", 3), ("waveout.csv", 4), ("pinch.csv", 5), ("open.csv", 1)]:
        df = pd.DataFrame(
            {
                "gt": np.full(150, cls_id, dtype=np.int64),
                **{f"emg_{c}": np.full(150, 16, dtype=np.int64) for c in range(8)},
            }
        )
        df.to_csv(epn_dir / name, index=False)

    npy_arr = np.zeros((150, 9), dtype=np.float32)
    npy_arr[:, 0] = 2
    npy_arr[:, 1:] = 32.0
    np.save(roam_dir / "close.npy", npy_arr)

    df_unlab = pd.DataFrame({f"emg_{c}": np.ones(150, dtype=np.int64) for c in range(8)})
    df_unlab.to_csv(roam_dir / "subject_unlabel.csv", index=False)

    emg_csv, gt_csv = load_reactemg_file(epn_dir / "wavein.csv", num_classes=6)
    assert emg_csv.shape == (150, 8) and np.all(gt_csv == 3)

    emg_npy, gt_npy = load_reactemg_file(roam_dir / "close.npy", num_classes=6)
    assert emg_npy.shape == (150, 8) and np.all(gt_npy == 2)

    splits_pub = discover_reactemg_files(tmp_path, dataset_selection="pub_with_epn", num_classes=6, val_split_ratio=0.2)
    all_supervised_names = {p.name for p in splits_pub["train"] + splits_pub["val"]}
    assert {"wavein.csv", "waveout.csv", "pinch.csv", "open.csv", "close.npy"}.issubset(all_supervised_names)
    assert all("unlabel" not in p.name for p in splits_pub["train"] + splits_pub["val"])
    assert len(splits_pub["unlabeled"]) == 1 and splits_pub["unlabeled"][0].name == "subject_unlabel.csv"

    splits_epn = discover_reactemg_files(tmp_path, dataset_selection="epn_only", num_classes=6, val_split_ratio=0.25)
    epn_names = {p.name for p in splits_epn["train"] + splits_epn["val"]}
    assert epn_names == {"wavein.csv", "waveout.csv", "pinch.csv", "open.csv"}


def test_reactemg_dataset_8tuple_and_sampler(tmp_path: Path) -> None:
    p = tmp_path / "seq.csv"
    gt = np.concatenate([np.zeros(60, dtype=np.int64), np.full(60, 5, dtype=np.int64)])
    df = pd.DataFrame({"gt": gt, **{f"emg_{c}": np.full(120, 64, dtype=np.int64) for c in range(8)}})
    df.to_csv(p, index=False)

    ds = ReactEMGDataset(
        [p],
        window_size=120,
        stride=10,
        num_classes=6,
        task_selection=(0, 1, 2, 3),
        deterministic=True,
        seed=12,
    )
    assert len(ds) == 4
    assert ds.labeled_indices == [0, 1, 2]
    assert ds.unlabeled_indices == [3]

    # Task 0: EMG clean, action masked
    emg_w, act_w, m_emg, m_act, mp_emg, mp_act, t_idx, trans_idx = ds[0]
    assert emg_w.shape == (120, 8) and m_emg.shape == (120, 8)
    assert act_w.shape == (120,) and m_act.shape == (120,)
    assert int(t_idx.item()) == 0
    assert int(trans_idx.item()) == 60
    assert not torch.any(mp_emg)
    assert torch.any(mp_act)

    # Task 1: Action clean, EMG masked
    _, _, _, _, mp_emg_1, mp_act_1, t_idx_1, _ = ds[1]
    assert int(t_idx_1.item()) == 1
    assert torch.any(mp_emg_1)
    assert not torch.any(mp_act_1)

    # Task 3: Self-supervised EMG (actions all <MASK> = 6, mask_pos_actions False)
    _, act_3, _, m_act_3, mp_emg_3, mp_act_3, t_idx_3, _ = ds[3]
    assert int(t_idx_3.item()) == 3
    assert torch.all(act_3 == 6) and torch.all(m_act_3 == 6)
    assert torch.any(mp_emg_3) and not torch.any(mp_act_3)

    sampler = LabeledUnlabeledSampler(ds, unlabeled_weight=0.5, seed=99)
    sampled = list(iter(sampler))
    assert len(sampled) == 4
    assert set(sampled) == {0, 1, 2, 3}

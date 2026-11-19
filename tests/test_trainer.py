"""Unit tests for ReactEMG metrics, look-ahead aggregation, and trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import ReactEMGDataset
from src.metrics import (
    LookaheadAggregator,
    evaluate_transition_accuracy,
    extract_buffer_windows,
    extract_maintenance_windows,
)
from src.model.model import Any2Any_Model
from src.trainer import ReactEMGTrainer, build_scheduler, process_batch


def test_lookahead_aggregator_online_and_sequence() -> None:
    agg = LookaheadAggregator(
        window_size=60,
        lookahead=10,
        samples_between_prediction=5,
        maj_vote_range="future",
        weighting_method="weighted",
        weight_max_factor=1.5,
        num_classes=6,
    )
    # Push logits strongly favoring class 4 (Wave-Out)
    logits = np.zeros((60, 6), dtype=np.float32)
    logits[:, 4] = 5.0
    for _ in range(15):
        pred, probs = agg.push_window_logits(logits)
    assert pred == 4
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert probs[4] > 0.9

    model = Any2Any_Model(num_classes=6, window_size=60, embedding_dim=32, nhead=4, num_layers=1)
    emg = np.random.default_rng(0).uniform(0.0, 0.5, size=(120, 8)).astype(np.float32)
    preds, probs_seq = agg.predict_sequence(model, emg, stride=2, batch_size=8, device="cpu")
    assert preds.shape == (120,)
    assert probs_seq.shape == (120, 6)
    assert np.all(preds[:59] == -1)
    assert np.all((preds[59:] >= 0) & (preds[59:] < 6))


def test_transition_and_maintenance_evaluation() -> None:
    gt = np.concatenate([np.zeros(250, dtype=np.int64), np.full(250, 5, dtype=np.int64)])
    buffers = extract_buffer_windows(gt, buffer_range=200)
    assert len(buffers) == 1
    assert buffers[0]["transition_idx"] == 250
    assert buffers[0]["start"] == 150 and buffers[0]["end"] == 350

    maint = extract_maintenance_windows(gt, buffer_range=200)
    assert len(maint) == 2

    # Clean switch 10 samples after gt transition (+50 ms at 200 Hz)
    preds_clean = np.concatenate([np.zeros(260, dtype=np.int64), np.full(240, 5, dtype=np.int64)])
    res_clean = evaluate_transition_accuracy(preds_clean, gt, buffer_range=200, fs=200)
    assert res_clean["transition_accuracy"] == 1.0
    assert np.isclose(res_clean["mean_offset_ms"], 50.0)
    assert res_clean["event_counts"]["passed"] == 1

    # Maintenance flicker outside transition buffer fails transition_accuracy
    preds_flicker = preds_clean.copy()
    preds_flicker[420] = 2
    res_flicker = evaluate_transition_accuracy(preds_flicker, gt, buffer_range=200, fs=200)
    assert res_flicker["transition_accuracy"] == 0.0
    assert res_flicker["transition_reasons"]["maintenance_flicker"] == 1


def test_trainer_process_batch_fit_and_resume(tmp_path: Path) -> None:
    model = Any2Any_Model(num_classes=6, window_size=60, embedding_dim=32, nhead=4, num_layers=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, total_steps=10, warmup_ratio=0.2, lr_decay_method="cosine")
    lrs = []
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < lrs[1] and lrs[-1] < lrs[2]

    csv_path = tmp_path / "rec.csv"
    gt = np.concatenate([np.zeros(60, dtype=np.int64), np.full(60, 2, dtype=np.int64)])
    df = pd.DataFrame({"gt": gt, **{f"emg_{c}": np.full(120, 48, dtype=np.int64) for c in range(8)}})
    df.to_csv(csv_path, index=False)

    ds = ReactEMGDataset([csv_path], window_size=60, stride=20, num_classes=6, task_selection=(0, 1, 2), seed=4)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    batch_res = process_batch(model, batch, device=torch.device("cpu"), num_classes=6)
    assert batch_res["loss"].requires_grad
    assert torch.isfinite(batch_res["loss"])

    trainer = ReactEMGTrainer(model, lr=5e-4, device="cpu")
    out_dir = tmp_path / "out"
    hist = trainer.fit(loader, val_loader=loader, epochs=1, output_dir=out_dir)
    assert len(hist) == 1
    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "history.csv").exists()

    # Resume for epoch 2
    trainer2 = ReactEMGTrainer(
        Any2Any_Model(num_classes=6, window_size=60, embedding_dim=32, nhead=4, num_layers=1),
        lr=5e-4,
        device="cpu",
    )
    hist2 = trainer2.fit(loader, val_loader=loader, epochs=2, output_dir=out_dir, resume_from=out_dir / "best_model.pt")
    assert len(hist2) == 1 and hist2[0]["epoch"] == 2.0

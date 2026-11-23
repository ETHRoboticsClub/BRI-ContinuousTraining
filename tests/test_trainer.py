"""Unit tests for WaveFormerTrainer, cosine warmup scheduler, metrics, and resume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import WaveFormerDataset
from src.model.waveformer import WaveFormer
from src.trainer import WaveFormerTrainer, build_cosine_warmup_scheduler


def test_cosine_warmup_scheduler_profile() -> None:
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_cosine_warmup_scheduler(opt, total_steps=10, warmup_steps=3, min_lr_ratio=0.01)
    lrs: list[float] = []
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < lrs[1] < lrs[2]
    assert lrs[-1] < lrs[2]


def test_waveformer_trainer_fit_resume_and_streaming_eval(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    files: list[Path] = []
    for cls_id in range(6):
        arr = np.zeros((300, 9), dtype=np.float32)
        arr[:, 1:] = rng.normal(0.0, 2.0, size=(300, 8))
        if cls_id > 0:
            arr[80:240, 0] = cls_id
            arr[80:240, 1 + cls_id] += 40.0 * np.sin(np.linspace(0, 20 * np.pi, 160))
        p = tmp_path / f"cls_{cls_id}.npy"
        np.save(p, arr)
        files.append(p)

    ds = WaveFormerDataset(files, window_size=80, channels=8, num_classes=6, train=True, seed=4)
    val_ds = WaveFormerDataset(files, window_size=80, channels=8, num_classes=6, train=False, seed=4)
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = WaveFormer(
        channels=8,
        num_classes=6,
        window_size=80,
        patch_size=(1, 10),
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        wt_levels=2,
    )
    trainer = WaveFormerTrainer(model, lr=1e-3, warmup_epochs=1, device="cpu")
    out_dir = tmp_path / "out"
    hist = trainer.fit(train_loader, val_loader=val_loader, epochs=2, output_dir=out_dir)
    assert len(hist) == 2
    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "history.csv").exists()

    stream_res = trainer.evaluate_streaming(files, window_size=80, step_size=20)
    assert stream_res["num_trials"] == 6
    assert 0.0 <= stream_res["streaming_trial_acc"] <= 1.0

    model2 = WaveFormer(
        channels=8,
        num_classes=6,
        window_size=80,
        patch_size=(1, 10),
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        wt_levels=2,
    )
    trainer2 = WaveFormerTrainer(model2, lr=1e-3, warmup_epochs=1, device="cpu")
    hist2 = trainer2.fit(
        train_loader, val_loader=val_loader, epochs=3, output_dir=out_dir, resume_from=out_dir / "best_model.pt"
    )
    assert len(hist2) == 1 and hist2[0]["epoch"] == 3.0

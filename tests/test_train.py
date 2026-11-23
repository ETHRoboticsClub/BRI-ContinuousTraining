"""End-to-end synthetic CLI integration tests for WaveFormer train and evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts import evaluate, train


def test_end_to_end_waveformer_train_and_evaluate(tmp_path: Path) -> None:
    data_root = tmp_path / "raw_reactemg"
    epn_root = data_root / "EMG-EPN-612"
    manifest_dir = tmp_path / "processed_waveformer"
    out_dir = tmp_path / "output"
    cfg_path = tmp_path / "params.yaml"

    rng = np.random.default_rng(123)
    for uid in ("trainingJSON_user1", "trainingJSON_user2", "trainingJSON_user3"):
        udir = epn_root / uid
        udir.mkdir(parents=True)
        for cls_id, name in enumerate(("noGesture", "open", "fist", "waveIn", "waveOut", "pinch")):
            arr = np.zeros((300, 9), dtype=np.float32)
            arr[:, 1:] = rng.normal(0.0, 2.0, size=(300, 8))
            if cls_id > 0:
                arr[80:240, 0] = cls_id
                arr[80:240, 1 + (cls_id % 8)] += 35.0 * np.sin(np.linspace(0, 16 * np.pi, 160))
            np.save(udir / f"trial_{name}.npy", arr)

    cfg = {
        "preprocessing": {
            "mode": "waveformer",
            "fs": 200,
            "channels": 8,
            "scale_divisor": 128.0,
            "hand_choice": "right",
            "remove_dc": True,
        },
        "model": {
            "arch": "waveformer",
            "num_classes": 6,
            "channels": 8,
            "input_channels": 1,
            "window_size": 100,
            "patch_size": [1, 10],
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "wt_levels": 2,
            "wt_type": "db1",
            "wt_kernel_size": 3,
            "highfreq_dropout": 0.1,
            "dropout": 0.1,
            "drop_path": 0.1,
        },
        "train": {
            "data_root": str(data_root),
            "cache_dir": str(manifest_dir),
            "dataset_selection": "epn_only",
            "epn_subset_percentage": 1.0,
            "window_stride": 40,
            "rescaling_range": [0.7, 1.4],
            "jitter_sigma": 0.004,
            "batch_size": 8,
            "epochs": 2,
            "lr": 0.001,
            "min_lr": 0.00001,
            "weight_decay": 0.01,
            "warmup_epochs": 1,
            "label_smoothing": 0.05,
            "grad_clip": 1.0,
            "val_split_ratio": 0.34,
            "seeds": [42],
            "device": "cpu",
        },
        "eval": {
            "window_size": 100,
            "step_size": 20,
            "max_files": 50,
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    manifest_file = manifest_dir / "missing_manifest.json"

    train.main(
        [
            "--config",
            str(cfg_path),
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ]
    )
    ckpt_path = out_dir / "best_model.pt"
    assert ckpt_path.exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "metrics.json").exists()

    metrics_path = out_dir / "eval_metrics.json"
    evaluate.main(
        [
            "--config",
            str(cfg_path),
            "--manifest",
            str(manifest_file),
            "--checkpoint",
            str(ckpt_path),
            "--output",
            str(metrics_path),
            "--device",
            "cpu",
        ]
    )
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["arch"] == "waveformer"
    assert "val_acc" in metrics and "streaming_trial_acc" in metrics

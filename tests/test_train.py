"""End-to-end synthetic CLI integration tests for build_dataset, train, and evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import build_dataset, evaluate, train


def test_end_to_end_synthetic_build_train_evaluate(tmp_path: Path) -> None:
    data_root = tmp_path / "raw_reactemg"
    manifest_dir = tmp_path / "processed_reactemg"
    out_dir = tmp_path / "output"
    cfg_path = tmp_path / "params.yaml"

    cfg = {
        "preprocessing": {
            "fs": 200,
            "channels": 8,
            "median_filter_size": 3,
            "medfilt_order": "before_rec",
            "scale_divisor": 128.0,
            "hand_choice": "right",
        },
        "model": {
            "num_classes": 6,
            "window_size": 60,
            "embedding_dim": 32,
            "nhead": 4,
            "num_layers": 1,
            "dropout": 0.1,
            "activation": "gelu",
            "embedding_method": "linear_projection",
            "mask_alignment": "aligned",
            "share_pe": True,
            "tie_weight": False,
            "use_input_layernorm": True,
            "output_reduction_method": "none",
            "chunk_size": 0,
        },
        "train": {
            "data_root": str(data_root),
            "dataset_selection": "pub_with_epn",
            "epn_subset_percentage": 1.0,
            "window_stride": 40,
            "task_selection": [0, 1, 2],
            "lambda_poisson": 7.0,
            "sampling_probability_poisson": 0.99,
            "poisson_mask_range": [0.3, 0.75],
            "end_mask_range": [0.3, 0.6],
            "transition_buffer": 10,
            "scale_emg_loss": 0,
            "noise": 0.0,
            "batch_size": 8,
            "epochs": 1,
            "lr": 0.0005,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "lr_decay_method": "linear",
            "grad_clip": 1.0,
            "saved_checkpoint_pth": None,
            "val_split_ratio": 0.34,
            "seeds": [42],
            "device": "cpu",
        },
        "eval": {
            "buffer_range": 40,
            "lookahead": 10,
            "samples_between_prediction": 10,
            "stride": 10,
            "maj_vote_range": "future",
            "likelihood_format": "probs",
            "weight_max_factor": 1.5,
            "weighting_method": "equal",
            "allow_relax": 0,
            "strict_transition": 0,
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 1. Build synthetic dataset + manifest
    build_dataset.main(
        [
            "--config",
            str(cfg_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(manifest_dir),
            "--synthetic-smoke",
        ]
    )
    manifest_file = manifest_dir / "manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["num_classes"] == 6
    assert manifest["counts"]["train"] >= 2
    assert manifest["counts"]["val"] >= 1
    assert manifest["counts"]["unlabeled"] == 1

    # 2. Train 1 epoch on synthetic manifest
    train.main(
        [
            "--config",
            str(cfg_path),
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--skip-streaming-eval",
        ]
    )
    ckpt_path = out_dir / "best_model.pt"
    assert ckpt_path.exists()
    assert (out_dir / "history.csv").exists()

    # 3. Evaluate with lookahead=0 and lookahead=10
    metrics_path = out_dir / "metrics.json"
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
        ]
    )
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "raw_accuracy" in metrics and "transition_accuracy" in metrics
    assert "lookahead_0" in metrics["by_lookahead"]
    assert "lookahead_10" in metrics["by_lookahead"]

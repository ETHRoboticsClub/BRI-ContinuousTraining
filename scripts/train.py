"""Train the 6-class WaveFormer sEMG gesture classifier on GPU (or CPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import (  # noqa: E402
    CLASS_NAMES_6,
    WaveFormerDataset,
    discover_reactemg_files,
)
from src.model.model import build_model_from_config  # noqa: E402
from src.trainer import (  # noqa: E402
    WaveFormerTrainer,
    load_waveformer_checkpoint,
    resolve_device,
)


def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_splits(cfg: dict, seed: int, manifest_path: Path | None) -> dict[str, list[Path]]:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_sel = (
            str(manifest.get("dataset_selection", "")).lower()
            == str(train_cfg.get("dataset_selection", "epn_only")).lower()
        )
        same_classes = int(manifest.get("num_classes", -1)) == int(model_cfg.get("num_classes", 6))
        same_pct = (
            abs(float(manifest.get("epn_subset_percentage", -1.0)) - float(train_cfg.get("epn_subset_percentage", 1.0)))
            < 1e-6
        )
        same_val = (
            abs(float(manifest.get("val_split_ratio", -1.0)) - float(train_cfg.get("val_split_ratio", 0.1))) < 1e-6
        )
        if same_sel and same_classes and same_pct and same_val:
            splits = manifest.get("splits", {})
            resolved = {
                k: [Path(p) for p in splits.get(k, []) if Path(p).exists()] for k in ("train", "val", "unlabeled")
            }
            if resolved["train"]:
                return resolved

    data_root = _resolve_path(train_cfg.get("data_root", "data/raw/reactemg"))
    return discover_reactemg_files(
        data_root=data_root,
        dataset_selection=str(train_cfg.get("dataset_selection", "epn_only")),
        num_classes=int(model_cfg.get("num_classes", 6)),
        epn_subset_percentage=float(train_cfg.get("epn_subset_percentage", 0.6)),
        val_split_ratio=float(train_cfg.get("val_split_ratio", 0.1)),
        seed=seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 6-class WaveFormer sEMG gesture classifier")
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "src" / "params.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "data" / "processed" / "waveformer" / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "output")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from output/best_model.pt")
    parser.add_argument("--skip-streaming-eval", action="store_true")
    return parser.parse_args(argv)


def train_from_config(args: argparse.Namespace) -> Path:
    """Execute WaveFormer training configured by params.yaml and CLI overrides."""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prep_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    seeds = [int(args.seed)] if args.seed is not None else [int(s) for s in train_cfg.get("seeds", [42])]
    seed = seeds[0]
    _set_seed(seed)

    epochs = int(args.epochs) if args.epochs is not None else int(train_cfg.get("epochs", 25))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 256))
    device = resolve_device(args.device if args.device is not None else str(train_cfg.get("device", "cuda")))
    output_dir = args.output_dir if args.output_dir.is_absolute() else (_REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_splits(cfg, seed=seed, manifest_path=args.manifest)
    max_files = train_cfg.get("max_files")
    if max_files is not None and int(max_files) > 0:
        splits["train"] = splits["train"][: int(max_files)]
    if not splits["train"]:
        raise RuntimeError(
            f"No training files found under {_resolve_path(train_cfg.get('data_root', 'data/raw/reactemg'))}."
        )

    channels = int(prep_cfg.get("channels", 8))
    num_classes = int(model_cfg.get("num_classes", 6))
    window_size = int(model_cfg.get("window_size", 200))
    scale_divisor = float(prep_cfg.get("scale_divisor", 128.0))
    hand_choice = str(prep_cfg.get("hand_choice", "right"))
    remove_dc = bool(prep_cfg.get("remove_dc", True))
    cache_dir_cfg = train_cfg.get("cache_dir")
    cache_dir = _resolve_path(cache_dir_cfg) if cache_dir_cfg else None

    rescaling_raw = train_cfg.get("rescaling_range", [0.55, 1.80])
    rescaling_range = (float(rescaling_raw[0]), float(rescaling_raw[1]))

    train_ds = WaveFormerDataset(
        file_paths=splits["train"],
        window_size=window_size,
        channels=channels,
        num_classes=num_classes,
        scale_divisor=scale_divisor,
        hand_choice=hand_choice,
        remove_dc=remove_dc,
        stride=int(train_cfg.get("window_stride", 40)),
        train=True,
        rescaling_range=rescaling_range,
        jitter_sigma=float(train_cfg.get("jitter_sigma", 0.004)),
        cache_dir=cache_dir,
        seed=seed,
    )

    counts = train_ds.class_counts()
    cls_map = dict(zip(CLASS_NAMES_6[:num_classes], counts, strict=True))
    print(f"Train windows={len(train_ds)} class_counts={cls_map}", flush=True)

    # Inverse-sqrt smoothed class weights prevent majority Relax bias without destabilizing gradients
    total_cnt = max(1, sum(counts))
    class_weights = [float((total_cnt / max(1, num_classes * c)) ** 0.5) if c > 0 else 1.0 for c in counts]
    mean_cw = float(np.mean(class_weights))
    class_weights = [w / max(1e-6, mean_cw) for w in class_weights]

    pin_mem = bool(device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)

    val_loader: DataLoader | None = None
    if splits["val"]:
        val_ds = WaveFormerDataset(
            file_paths=splits["val"],
            window_size=window_size,
            channels=channels,
            num_classes=num_classes,
            scale_divisor=scale_divisor,
            hand_choice=hand_choice,
            remove_dc=remove_dc,
            stride=int(train_cfg.get("window_stride", 40)),
            train=False,
            cache_dir=cache_dir,
            seed=seed,
        )
        val_cls_map = dict(zip(CLASS_NAMES_6[:num_classes], val_ds.class_counts(), strict=True))
        print(f"Val windows={len(val_ds)} class_counts={val_cls_map}", flush=True)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    saved_ckpt = train_cfg.get("saved_checkpoint_pth")
    if saved_ckpt is not None:
        model, _ = load_waveformer_checkpoint(_resolve_path(saved_ckpt), device=device)
    else:
        model = build_model_from_config({**model_cfg, "arch": "waveformer"}, channels=channels)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"WaveFormer initialized on {device} | total_params={total_params:,}", flush=True)

    trainer = WaveFormerTrainer(
        model=model,
        lr=float(train_cfg.get("lr", 5e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 3)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.05)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        class_weights=class_weights,
        device=device,
    )

    resume_path = output_dir / "best_model.pt" if args.resume and (output_dir / "best_model.pt").exists() else None
    checkpoint_args = {**model_cfg, **prep_cfg, "arch": "waveformer", "channels": channels}

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        output_dir=output_dir,
        checkpoint_args=checkpoint_args,
        resume_from=resume_path,
    )
    print(f"Training complete ({len(history)} epochs on {device}). Best checkpoint: {output_dir / 'best_model.pt'}")

    # Reload best weights for final metrics report & streaming simulation
    best_model, best_ckpt = load_waveformer_checkpoint(output_dir / "best_model.pt", device=device)
    trainer.model = best_model
    report: dict[str, Any] = {
        "arch": "waveformer",
        "device": str(device),
        "epochs": len(history),
        "best_epoch": int(best_ckpt.get("epoch", len(history))),
        "val_metrics": best_ckpt.get("metrics", {}),
    }

    if not args.skip_streaming_eval and splits["val"]:
        stream_metrics = trainer.evaluate_streaming(
            val_files=splits["val"],
            window_size=window_size,
            step_size=int(eval_cfg.get("step_size", 20)),
            scale_divisor=scale_divisor,
            hand_choice=hand_choice,
            remove_dc=remove_dc,
            max_files=int(eval_cfg.get("max_files", 600)),
        )
        report["streaming_eval"] = stream_metrics
        print(
            f"Streaming eval ({stream_metrics['num_trials']} unseen-subject trials): "
            f"step_acc={stream_metrics['streaming_step_acc']:.4f}, "
            f"bacc={stream_metrics['streaming_balanced_acc']:.4f}, "
            f"macro_f1={stream_metrics['streaming_macro_f1']:.4f}, "
            f"trial_acc={stream_metrics['streaming_trial_acc']:.4f}",
            flush=True,
        )

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_dir / "best_model.pt"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_from_config(args)


if __name__ == "__main__":
    main()

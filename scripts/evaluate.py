"""Evaluate a trained WaveFormer checkpoint on unseen subjects & causal 10 Hz streaming."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import WaveFormerDataset, discover_reactemg_files  # noqa: E402
from src.trainer import WaveFormerTrainer, load_waveformer_checkpoint, resolve_device  # noqa: E402


def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained WaveFormer checkpoint")
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "src" / "params.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "data" / "processed" / "waveformer" / "manifest.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=_REPO_ROOT / "output" / "best_model.pt")
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "output" / "metrics.json")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def evaluate_from_config(args: argparse.Namespace) -> dict[str, object]:
    """Load WaveFormer checkpoint, compute validation metrics & streaming trial accuracy."""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prep_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    device = resolve_device(args.device if args.device is not None else str(train_cfg.get("device", "auto")))
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (_REPO_ROOT / args.checkpoint)
    model, _ = load_waveformer_checkpoint(ckpt_path, device=device)

    val_files: list[Path] = []
    if args.manifest is not None and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        val_files = [Path(p) for p in manifest.get("splits", {}).get("val", []) if Path(p).exists()]

    if not val_files:
        splits = discover_reactemg_files(
            data_root=_resolve_path(train_cfg.get("data_root", "data/raw/reactemg")),
            dataset_selection=str(train_cfg.get("dataset_selection", "epn_only")),
            num_classes=int(model_cfg.get("num_classes", 6)),
            epn_subset_percentage=float(train_cfg.get("epn_subset_percentage", 0.6)),
            val_split_ratio=float(train_cfg.get("val_split_ratio", 0.1)),
        )
        val_files = splits["val"] or splits["train"]

    channels = int(prep_cfg.get("channels", 8))
    num_classes = int(model_cfg.get("num_classes", 6))
    window_size = int(model_cfg.get("window_size", 200))
    scale_divisor = float(prep_cfg.get("scale_divisor", 128.0))
    hand_choice = str(prep_cfg.get("hand_choice", "right"))
    remove_dc = bool(prep_cfg.get("remove_dc", True))

    val_ds = WaveFormerDataset(
        file_paths=val_files,
        window_size=window_size,
        channels=channels,
        num_classes=num_classes,
        scale_divisor=scale_divisor,
        hand_choice=hand_choice,
        remove_dc=remove_dc,
        stride=int(train_cfg.get("window_stride", 40)),
        train=False,
    )
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.get("batch_size", 256)), shuffle=False)

    trainer = WaveFormerTrainer(model=model, device=device)
    val_metrics = trainer.validate_epoch(val_loader)
    stream_metrics = trainer.evaluate_streaming(
        val_files=val_files,
        window_size=window_size,
        step_size=int(eval_cfg.get("step_size", 20)),
        scale_divisor=scale_divisor,
        hand_choice=hand_choice,
        remove_dc=remove_dc,
        max_files=int(eval_cfg.get("max_files", 600)),
    )

    report: dict[str, object] = {
        "arch": "waveformer",
        "checkpoint": str(ckpt_path),
        "num_val_files": len(val_files),
        "val_metrics": val_metrics,
        "streaming_eval": stream_metrics,
        "val_acc": val_metrics["val_acc"],
        "val_balanced_acc": val_metrics["val_balanced_acc"],
        "val_macro_f1": val_metrics["val_macro_f1"],
        "streaming_trial_acc": stream_metrics["streaming_trial_acc"],
    }

    out_path = args.output if args.output.is_absolute() else (_REPO_ROOT / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation metrics written to {out_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_from_config(args)


if __name__ == "__main__":
    main()

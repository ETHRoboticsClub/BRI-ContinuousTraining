"""Evaluate a trained ReactEMG checkpoint under zero-lookahead and smoothed look-ahead modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import discover_reactemg_files  # noqa: E402
from src.preprocess import Preprocess  # noqa: E402
from src.trainer import ReactEMGTrainer, load_reactemg_checkpoint, resolve_device  # noqa: E402


def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ReactEMG checkpoint")
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "src" / "params.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=_REPO_ROOT / "data" / "processed" / "reactemg" / "manifest.json"
    )
    parser.add_argument("--checkpoint", type=Path, default=_REPO_ROOT / "output" / "best_model.pt")
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "output" / "metrics.json")
    parser.add_argument(
        "--lookahead", type=int, default=None, help="Override single lookahead horizon instead of [0, eval.lookahead]"
    )
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def evaluate_from_config(args: argparse.Namespace) -> dict[str, object]:
    """Load checkpoint and write zero-lookahead + smoothed look-ahead validation metrics."""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prep_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    device = resolve_device(args.device if args.device is not None else str(train_cfg.get("device", "auto")))
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (_REPO_ROOT / args.checkpoint)
    model, _ = load_reactemg_checkpoint(ckpt_path, device=device)

    val_files: list[Path] = []
    if args.manifest is not None and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        val_files = [Path(p) for p in manifest.get("splits", {}).get("val", []) if Path(p).exists()]

    if not val_files:
        splits = discover_reactemg_files(
            data_root=_resolve_path(train_cfg.get("data_root", "data/raw/reactemg")),
            dataset_selection=str(train_cfg.get("dataset_selection", "pub_with_epn")),
            num_classes=int(model_cfg.get("num_classes", 6)),
            epn_subset_percentage=float(train_cfg.get("epn_subset_percentage", 1.0)),
            val_split_ratio=float(train_cfg.get("val_split_ratio", 0.05)),
        )
        val_files = splits["val"] or splits["train"]

    preprocess = Preprocess(
        fs=int(prep_cfg.get("fs", 200)),
        channels=int(prep_cfg.get("channels", 8)),
        median_filter_size=int(prep_cfg.get("median_filter_size", 3)),
        medfilt_order=str(prep_cfg.get("medfilt_order", "before_rec")),
        scale_divisor=float(prep_cfg.get("scale_divisor", 128.0)),
        hand_choice=str(prep_cfg.get("hand_choice", "right")),
    )

    trainer = ReactEMGTrainer(model=model, device=device)
    configured_lookahead = int(eval_cfg.get("lookahead", 50))
    horizons = [int(args.lookahead)] if args.lookahead is not None else sorted({0, configured_lookahead})
    stride = int(args.stride) if args.stride is not None else int(eval_cfg.get("stride", 1))

    by_lookahead: dict[str, object] = {}
    for la in horizons:
        by_lookahead[f"lookahead_{la}"] = trainer.evaluate_streaming(
            val_files=val_files,
            preprocess=preprocess,
            lookahead=la,
            samples_between_prediction=int(eval_cfg.get("samples_between_prediction", 20)),
            stride=stride,
            buffer_range=int(eval_cfg.get("buffer_range", 200)),
            maj_vote_range=str(eval_cfg.get("maj_vote_range", "future")),
            likelihood_format=str(eval_cfg.get("likelihood_format", "probs")),
            weight_max_factor=float(eval_cfg.get("weight_max_factor", 1.5)),
            weighting_method=str(eval_cfg.get("weighting_method", "equal")),
            allow_relax=int(eval_cfg.get("allow_relax", 0)),
            strict_transition=int(eval_cfg.get("strict_transition", 0)),
            fs=int(prep_cfg.get("fs", 200)),
        )

    primary_key = f"lookahead_{configured_lookahead}" if args.lookahead is None else f"lookahead_{args.lookahead}"
    primary = dict(by_lookahead[primary_key])  # type: ignore[arg-type]
    report: dict[str, object] = {
        **primary,
        "checkpoint": str(ckpt_path),
        "num_val_files": len(val_files),
        "by_lookahead": by_lookahead,
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

"""Train the 6-class ReactEMG masked multimodal Transformer classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import LabeledUnlabeledSampler, ReactEMGDataset, discover_reactemg_files  # noqa: E402
from src.model.model import Any2Any_Model  # noqa: E402
from src.preprocess import Preprocess  # noqa: E402
from src.trainer import ReactEMGTrainer, load_reactemg_checkpoint, resolve_device  # noqa: E402


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
        if same_sel and same_classes and same_pct:
            splits = manifest.get("splits", {})
            resolved = {
                k: [Path(p) for p in splits.get(k, []) if Path(p).exists()] for k in ("train", "val", "unlabeled")
            }
            if resolved["train"]:
                return resolved

    data_root = _resolve_path(train_cfg.get("data_root", "data/raw/reactemg"))
    return discover_reactemg_files(
        data_root=data_root,
        dataset_selection=str(train_cfg.get("dataset_selection", "pub_with_epn")),
        num_classes=int(model_cfg.get("num_classes", 6)),
        epn_subset_percentage=float(train_cfg.get("epn_subset_percentage", 1.0)),
        val_split_ratio=float(train_cfg.get("val_split_ratio", 0.05)),
        seed=seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 6-class ReactEMG Any2Any_Model")
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "src" / "params.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=_REPO_ROOT / "data" / "processed" / "reactemg" / "manifest.json"
    )
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "output")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from saved_checkpoint_pth or output/best_model.pt"
    )
    parser.add_argument("--skip-streaming-eval", action="store_true")
    return parser.parse_args(argv)


def train_from_config(args: argparse.Namespace) -> Path:
    """Execute training configured by params.yaml and CLI overrides."""
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prep_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    seeds = [int(args.seed)] if args.seed is not None else [int(s) for s in train_cfg.get("seeds", [42])]
    seed = seeds[0]
    _set_seed(seed)

    epochs = int(args.epochs) if args.epochs is not None else int(train_cfg.get("epochs", 12))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 128))
    device = resolve_device(args.device if args.device is not None else str(train_cfg.get("device", "auto")))
    output_dir = args.output_dir if args.output_dir.is_absolute() else (_REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_splits(cfg, seed=seed, manifest_path=args.manifest)
    max_files = train_cfg.get("max_files")
    if max_files is not None and int(max_files) > 0:
        splits["train"] = splits["train"][: int(max_files)]
    if not splits["train"]:
        raise RuntimeError(
            f"No training files found under {_resolve_path(train_cfg.get('data_root', 'data/raw/reactemg'))}. "
            "Run scripts/build_dataset.py first."
        )

    preprocess = Preprocess(
        fs=int(prep_cfg.get("fs", 200)),
        channels=int(prep_cfg.get("channels", 8)),
        median_filter_size=int(prep_cfg.get("median_filter_size", 3)),
        medfilt_order=str(prep_cfg.get("medfilt_order", "before_rec")),
        scale_divisor=float(prep_cfg.get("scale_divisor", 128.0)),
        hand_choice=str(prep_cfg.get("hand_choice", "right")),
    )

    channels = int(prep_cfg.get("channels", 8))
    num_classes = int(model_cfg.get("num_classes", 6))
    window_size = int(model_cfg.get("window_size", 600))
    task_selection = [int(t) for t in train_cfg.get("task_selection", [0, 1, 2])]
    noise_amp = float(train_cfg.get("noise_std", train_cfg.get("noise", 0.0)))
    with_curr = bool(train_cfg.get("with_training_curriculum", False))
    emb_method = str(model_cfg.get("embedding_method", "linear_projection"))
    stage_1_w = tuple(float(x) for x in train_cfg.get("stage_1_weights", [0.5, 0.5]))
    stage_2_w = tuple(float(x) for x in train_cfg.get("stage_2_weights", [0.01, 0.99]))
    trans_buf = int(train_cfg.get("transition_buffer", 50))

    labeled_train_ds = ReactEMGDataset(
        file_paths=splits["train"],
        window_size=window_size,
        stride=int(train_cfg.get("window_stride", 30)),
        channels=channels,
        num_classes=num_classes,
        task_selection=[t for t in task_selection if t in (0, 1, 2)],
        preprocess=preprocess,
        mask_alignment=str(model_cfg.get("mask_alignment", "aligned")),
        lambda_poisson=float(train_cfg.get("lambda_poisson", 7.0)),
        sampling_probability_poisson=float(train_cfg.get("sampling_probability_poisson", 0.99)),
        poisson_mask_range=tuple(train_cfg.get("poisson_mask_range", [0.3, 0.75])),
        end_mask_range=tuple(train_cfg.get("end_mask_range", [0.3, 0.6])),
        transition_buffer=trans_buf,
        stage_1_weights=stage_1_w,
        stage_2_weights=stage_2_w,
        with_training_curriculum=with_curr,
        embedding_method=emb_method,
        noise=noise_amp,
        is_unlabeled=False,
        deterministic=False,
        seed=seed,
    )

    pin_mem = bool(device.type == "cuda")
    if 3 in task_selection and splits["unlabeled"]:
        unlabeled_ds = ReactEMGDataset(
            file_paths=splits["unlabeled"],
            window_size=window_size,
            stride=int(train_cfg.get("window_stride", 30)),
            channels=channels,
            num_classes=num_classes,
            task_selection=(3,),
            preprocess=preprocess,
            mask_alignment=str(model_cfg.get("mask_alignment", "aligned")),
            lambda_poisson=float(train_cfg.get("lambda_poisson", 7.0)),
            poisson_mask_range=tuple(train_cfg.get("poisson_mask_range", [0.3, 0.75])),
            transition_buffer=trans_buf,
            with_training_curriculum=with_curr,
            embedding_method=emb_method,
            is_unlabeled=True,
            deterministic=False,
            seed=seed,
        )
        if len(unlabeled_ds) > 0:
            labeled_count = len(labeled_train_ds)
            combined = ConcatDataset([labeled_train_ds, unlabeled_ds])
            # Expose index lists matching ConcatDataset offsets for LabeledUnlabeledSampler
            combined.labeled_indices = list(range(labeled_count))  # type: ignore[attr-defined]
            combined.unlabeled_indices = list(range(labeled_count, labeled_count + len(unlabeled_ds)))  # type: ignore[attr-defined]
            sampler = LabeledUnlabeledSampler(combined, seed=seed)  # type: ignore[arg-type]
            train_loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, pin_memory=pin_mem)
        else:
            train_loader = DataLoader(labeled_train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(labeled_train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)

    val_loader: DataLoader | None = None
    if splits["val"]:
        val_ds = ReactEMGDataset(
            file_paths=splits["val"],
            window_size=window_size,
            stride=int(train_cfg.get("window_stride", 30)),
            channels=channels,
            num_classes=num_classes,
            task_selection=(0,),
            preprocess=preprocess,
            mask_alignment=str(model_cfg.get("mask_alignment", "aligned")),
            lambda_poisson=float(train_cfg.get("lambda_poisson", 7.0)),
            sampling_probability_poisson=1.0,
            poisson_mask_range=tuple(train_cfg.get("poisson_mask_range", [0.3, 0.75])),
            transition_buffer=trans_buf,
            stage_1_weights=stage_1_w,
            stage_2_weights=stage_2_w,
            with_training_curriculum=with_curr,
            embedding_method=emb_method,
            is_unlabeled=False,
            deterministic=True,
            seed=seed,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    saved_ckpt = train_cfg.get("saved_checkpoint_pth")
    if saved_ckpt is not None:
        model, _ = load_reactemg_checkpoint(_resolve_path(saved_ckpt), device=device)
    else:
        model = Any2Any_Model(
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            nhead=int(model_cfg.get("nhead", 4)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            activation=str(model_cfg.get("activation", "gelu")),
            num_layers=int(model_cfg.get("num_layers", 2)),
            window_size=window_size,
            embedding_method=emb_method,
            mask_alignment=str(model_cfg.get("mask_alignment", "aligned")),
            share_pe=bool(model_cfg.get("share_pe", True)),
            tie_weight=bool(model_cfg.get("tie_weight", False)),
            use_input_layernorm=bool(model_cfg.get("use_input_layernorm", True)),
            num_classes=num_classes,
            output_reduction_method=str(model_cfg.get("output_reduction_method", "none")),
            chunk_size=int(model_cfg.get("chunk_size", 0)),
            channels=channels,
        )

    use_warmup = bool(train_cfg.get("use_warmup_and_decay", True))
    trainer = ReactEMGTrainer(
        model=model,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)) if use_warmup else 0.0,
        lr_decay_method=str(train_cfg.get("lr_decay_method", "linear")),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        scale_emg_loss=float(train_cfg.get("scale_emg_loss", 0)),
        device=device,
    )

    resume_path = output_dir / "best_model.pt" if args.resume and (output_dir / "best_model.pt").exists() else None
    checkpoint_args = {**model_cfg, **prep_cfg, "channels": channels}

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        output_dir=output_dir,
        checkpoint_args=checkpoint_args,
        resume_from=resume_path,
    )
    print(f"Training complete ({len(history)} epochs on {device}). Best checkpoint: {output_dir / 'best_model.pt'}")

    if not args.skip_streaming_eval and splits["val"]:
        metrics = trainer.evaluate_streaming(
            val_files=splits["val"],
            preprocess=preprocess,
            lookahead=int(eval_cfg.get("lookahead", 50)),
            samples_between_prediction=int(eval_cfg.get("samples_between_prediction", 20)),
            stride=int(eval_cfg.get("stride", 1)),
            buffer_range=int(eval_cfg.get("buffer_range", 200)),
            maj_vote_range=str(eval_cfg.get("maj_vote_range", "future")),
            likelihood_format=str(eval_cfg.get("likelihood_format", "probs")),
            weight_max_factor=float(eval_cfg.get("weight_max_factor", 1.5)),
            weighting_method=str(eval_cfg.get("weighting_method", "equal")),
            allow_relax=int(eval_cfg.get("allow_relax", 0)),
            strict_transition=int(eval_cfg.get("strict_transition", 0)),
            fs=int(prep_cfg.get("fs", 200)),
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return output_dir / "best_model.pt"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_from_config(args)


if __name__ == "__main__":
    main()

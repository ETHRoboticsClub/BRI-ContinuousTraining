"""ReactEMG multi-task masked trainer, scheduler, and checkpoint utilities."""

from __future__ import annotations

import csv
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import load_reactemg_file
from src.metrics import LookaheadAggregator, aggregate_sequence_metrics, evaluate_transition_accuracy
from src.model.model import Any2Any_Model
from src.preprocess import Preprocess


def resolve_device(device_pref: str | None = None) -> torch.device:
    """Resolve PyTorch device cleanly with CUDA auto-detection and CPU fallback."""
    pref = (device_pref or "auto").lower()
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    lr_decay_method: str = "linear",
) -> LambdaLR:
    """Create warmup + decay LambdaLR matching roamlab/reactemg."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(round(total_steps * float(warmup_ratio))))
    decay_method = str(lr_decay_method).lower()

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps + 1)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if decay_method == "linear":
            return max(0.0, 1.0 - progress)
        if decay_method == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if decay_method == "exponential":
            return max(0.0, 0.1**progress)
        raise ValueError(f"Unsupported lr_decay_method {lr_decay_method!r}")

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def process_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    scale_emg_loss: float | int = 0,
    num_classes: int = 6,
) -> dict[str, Any]:
    """Run ReactEMG masked multimodal forward pass and compute masked MSE + CE losses."""
    (
        emg_window,
        action_window,
        masked_emg,
        masked_actions,
        mask_positions_emg,
        mask_positions_actions,
        task_idx,
        _transition_idx,
    ) = batch
    _ = masked_emg

    emg_window = emg_window.to(device=device, dtype=torch.float32)
    action_window = action_window.to(device=device, dtype=torch.long)
    masked_actions = masked_actions.to(device=device, dtype=torch.long)
    mask_positions_emg = mask_positions_emg.to(device=device, dtype=torch.bool)
    mask_positions_actions = mask_positions_actions.to(device=device, dtype=torch.bool)
    task_idx = task_idx.to(device=device, dtype=torch.long)

    # Pass raw emg_window: Any2Any_Model replaces mask_positions_emg after linear projection.
    emg_out, action_out = model(emg_window, masked_actions, task_idx, mask_positions_emg)

    emg_count = int(mask_positions_emg.sum().item())
    if emg_count > 0:
        emg_loss = F.mse_loss(emg_out[mask_positions_emg], emg_window[mask_positions_emg], reduction="mean")
    else:
        emg_loss = torch.zeros((), device=device, dtype=torch.float32)

    valid_action_mask = mask_positions_actions & (action_window >= 0) & (action_window < int(num_classes))
    action_count = int(valid_action_mask.sum().item())
    if action_count > 0:
        masked_logits = action_out[valid_action_mask]
        masked_targets = action_window[valid_action_mask]
        action_loss = F.cross_entropy(masked_logits, masked_targets, reduction="mean")
        preds = torch.argmax(masked_logits[:, : int(num_classes)], dim=-1)
        action_correct = int((preds == masked_targets).sum().item())
    else:
        action_loss = torch.zeros((), device=device, dtype=torch.float32)
        action_correct = 0

    scale_val = float(scale_emg_loss)
    if scale_val == 1.0:
        emg_weight = 100.0
    elif scale_val > 1.0:
        emg_weight = scale_val
    else:
        emg_weight = 1.0

    if emg_count > 0 and action_count > 0:
        total_loss = emg_weight * emg_loss + action_loss
    elif emg_count > 0:
        total_loss = emg_weight * emg_loss
    elif action_count > 0:
        total_loss = action_loss
    else:
        # Keep computational graph attached on degenerate all-unmasked edge batch
        total_loss = 0.0 * (emg_out.sum() + action_out.sum())

    total_mask_positions = int(mask_positions_emg.sum().item() + mask_positions_actions.sum().item())

    return {
        "loss": total_loss,
        "emg_loss": float(emg_loss.detach().item()),
        "action_loss": float(action_loss.detach().item()),
        "emg_count": emg_count,
        "action_count": action_count,
        "action_correct": action_correct,
        "total_mask_positions": total_mask_positions,
    }


def save_reactemg_checkpoint(
    path: str | Path,
    model: Any2Any_Model,
    args_dict: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
) -> Path:
    """Save checkpoint in roamlab/reactemg-compatible dictionary layout."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_info": {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        "args_dict": dict(args_dict),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
    }
    torch.save(payload, out_path)
    return out_path


def load_reactemg_checkpoint(
    path: str | Path,
    model: Any2Any_Model | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Any2Any_Model, dict[str, Any]]:
    """Load checkpoint written by ReactEMG / BRI-TrainingPipeline."""
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    args_dict = dict(ckpt.get("args_dict", {})) if isinstance(ckpt, dict) else {}

    if model is None:
        model = Any2Any_Model(
            embedding_dim=int(args_dict.get("embedding_dim", 128)),
            nhead=int(args_dict.get("nhead", 4)),
            dropout=float(args_dict.get("dropout", 0.15)),
            activation=str(args_dict.get("activation", "gelu")),
            num_layers=int(args_dict.get("num_layers", 2)),
            window_size=int(args_dict.get("window_size", 600)),
            embedding_method=str(args_dict.get("embedding_method", "linear_projection")),
            mask_alignment=str(args_dict.get("mask_alignment", "aligned")),
            share_pe=bool(args_dict.get("share_pe", True)),
            tie_weight=bool(args_dict.get("tie_weight", False)),
            use_input_layernorm=bool(args_dict.get("use_input_layernorm", True)),
            num_classes=int(args_dict.get("num_classes", 6)),
            output_reduction_method=str(args_dict.get("output_reduction_method", "none")),
            chunk_size=int(args_dict.get("chunk_size", 0)),
            channels=int(args_dict.get("channels", 8)),
        )

    if isinstance(ckpt, dict) and "model_info" in ckpt and "model_state_dict" in ckpt["model_info"]:
        state_dict = ckpt["model_info"]["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(dev)
    return model, ckpt if isinstance(ckpt, dict) else {}


class ReactEMGTrainer:
    """Handles optimization, LR warmup/decay, validation, and checkpointing."""

    def __init__(
        self,
        model: Any2Any_Model,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        lr_decay_method: str = "linear",
        grad_clip: float = 1.0,
        scale_emg_loss: float | int = 0,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = device if isinstance(device, torch.device) else resolve_device(str(device))
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.lr_decay_method = str(lr_decay_method)
        self.grad_clip = float(grad_clip)
        self.scale_emg_loss = float(scale_emg_loss)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler: LambdaLR | None = None
        self._use_bf16 = bool(self.device.type == "cuda" and torch.cuda.is_bf16_supported())

    def _autocast_ctx(self):
        if self._use_bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Run one training epoch weighted by masked EMG and action counts."""
        self.model.train()
        total_loss_sum = 0.0
        total_weight_sum = 0
        emg_loss_weighted = 0.0
        emg_count_total = 0
        act_loss_weighted = 0.0
        act_count_total = 0
        act_correct_total = 0

        iterator = tqdm(train_loader, desc="Train", leave=False, disable=len(train_loader) <= 20)
        for batch in iterator:
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_ctx():
                res = process_batch(
                    self.model,
                    batch,
                    device=self.device,
                    scale_emg_loss=self.scale_emg_loss,
                    num_classes=self.model.num_classes,
                )
            loss = res["loss"]
            loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_weight = res["total_mask_positions"] if res["total_mask_positions"] > 0 else int(batch[0].size(0))
            total_loss_sum += float(loss.detach().item()) * batch_weight
            total_weight_sum += batch_weight
            if res["emg_count"] > 0:
                emg_loss_weighted += res["emg_loss"] * res["emg_count"]
                emg_count_total += res["emg_count"]
            if res["action_count"] > 0:
                act_loss_weighted += res["action_loss"] * res["action_count"]
                act_count_total += res["action_count"]
                act_correct_total += res["action_correct"]

        return {
            "train_loss": total_loss_sum / max(1, total_weight_sum),
            "train_emg_loss": emg_loss_weighted / max(1, emg_count_total),
            "train_action_loss": act_loss_weighted / max(1, act_count_total),
            "train_action_acc": act_correct_total / max(1, act_count_total),
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation pass weighted accurately by masked token counts."""
        self.model.eval()
        total_loss_sum = 0.0
        total_weight_sum = 0
        emg_loss_weighted = 0.0
        emg_count_total = 0
        act_loss_weighted = 0.0
        act_count_total = 0
        act_correct_total = 0

        for batch in val_loader:
            with self._autocast_ctx():
                res = process_batch(
                    self.model,
                    batch,
                    device=self.device,
                    scale_emg_loss=self.scale_emg_loss,
                    num_classes=self.model.num_classes,
                )
            batch_weight = res["total_mask_positions"] if res["total_mask_positions"] > 0 else int(batch[0].size(0))
            total_loss_sum += float(res["loss"].detach().item()) * batch_weight
            total_weight_sum += batch_weight
            if res["emg_count"] > 0:
                emg_loss_weighted += res["emg_loss"] * res["emg_count"]
                emg_count_total += res["emg_count"]
            if res["action_count"] > 0:
                act_loss_weighted += res["action_loss"] * res["action_count"]
                act_count_total += res["action_count"]
                act_correct_total += res["action_correct"]

        return {
            "val_loss": total_loss_sum / max(1, total_weight_sum),
            "val_emg_loss": emg_loss_weighted / max(1, emg_count_total),
            "val_action_loss": act_loss_weighted / max(1, act_count_total),
            "val_action_acc": act_correct_total / max(1, act_count_total),
        }

    @torch.no_grad()
    def evaluate_streaming(
        self,
        val_files: Sequence[str | Path],
        preprocess: Preprocess | None = None,
        lookahead: int = 50,
        samples_between_prediction: int = 20,
        stride: int = 1,
        buffer_range: int = 200,
        maj_vote_range: str = "future",
        likelihood_format: str = "probs",
        weight_max_factor: float = 1.5,
        weighting_method: str = "equal",
        allow_relax: int = 0,
        strict_transition: int = 0,
        fs: int = 200,
    ) -> dict[str, object]:
        """Run online sliding-window look-ahead evaluation across held-out files."""
        aggregator = LookaheadAggregator(
            window_size=self.model.window_size,
            lookahead=lookahead,
            samples_between_prediction=samples_between_prediction,
            maj_vote_range=maj_vote_range,
            likelihood_format=likelihood_format,
            weight_max_factor=weight_max_factor,
            weighting_method=weighting_method,
            num_classes=self.model.num_classes,
        )
        per_file_results: list[dict[str, object]] = []
        for path in val_files:
            emg_tc, gt = load_reactemg_file(
                path,
                preprocess=preprocess,
                channels=self.model.channels,
                num_classes=self.model.num_classes,
            )
            if len(emg_tc) < self.model.window_size or np.all(gt < 0):
                continue
            preds, _ = aggregator.predict_sequence(
                self.model, emg_tc, stride=stride, batch_size=128, device=self.device, pad_initial=True
            )
            res = evaluate_transition_accuracy(
                preds,
                gt,
                buffer_range=buffer_range,
                fs=fs,
                allow_relax=allow_relax,
                strict_transition=strict_transition,
            )
            per_file_results.append(res)
        return aggregate_sequence_metrics(per_file_results)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 12,
        output_dir: str | Path = "output",
        checkpoint_args: dict[str, Any] | None = None,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Train across epochs, saving best_model.pt and history.csv."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = out_dir / "best_model.pt"

        total_steps = max(1, int(epochs) * max(1, len(train_loader)))
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=self.warmup_ratio,
            lr_decay_method=self.lr_decay_method,
        )

        start_epoch = 0
        best_val_loss = float("inf")
        if resume_from is not None and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            model_info = ckpt.get("model_info", {})
            if "model_state_dict" in model_info:
                self.model.load_state_dict(model_info["model_state_dict"])
            if model_info.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(model_info["optimizer_state_dict"])
            if model_info.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(model_info["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

        args_to_save = dict(checkpoint_args or {})
        args_to_save.setdefault("embedding_dim", getattr(self.model, "embedding_dim", 128))
        args_to_save.setdefault("nhead", getattr(self.model, "nhead", 4))
        args_to_save.setdefault("dropout", getattr(self.model, "dropout", 0.15))
        args_to_save.setdefault("activation", getattr(self.model, "activation", "gelu"))
        args_to_save.setdefault("num_layers", getattr(self.model, "num_layers", 2))
        args_to_save.setdefault("window_size", getattr(self.model, "window_size", 600))
        args_to_save.setdefault("embedding_method", getattr(self.model, "embedding_method", "linear_projection"))
        args_to_save.setdefault("mask_alignment", getattr(self.model, "mask_alignment", "aligned"))
        args_to_save.setdefault("share_pe", getattr(self.model, "share_pe", True))
        args_to_save.setdefault("tie_weight", getattr(self.model, "tie_weight", False))
        args_to_save.setdefault("use_input_layernorm", getattr(self.model, "use_input_layernorm", True))
        args_to_save.setdefault("num_classes", getattr(self.model, "num_classes", 6))
        args_to_save.setdefault("output_reduction_method", getattr(self.model, "output_reduction_method", "none"))
        args_to_save.setdefault("chunk_size", getattr(self.model, "chunk_size", 0))
        args_to_save.setdefault("channels", getattr(self.model, "channels", 8))

        history: list[dict[str, float]] = []
        for epoch_idx in range(start_epoch, int(epochs)):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch_idx)
            for ds in (
                getattr(train_loader, "dataset", None),
                getattr(val_loader, "dataset", None) if val_loader is not None else None,
            ):
                if ds is None:
                    continue
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(epoch_idx, int(epochs))
                for sub_ds in getattr(ds, "datasets", ()):
                    if hasattr(sub_ds, "set_epoch"):
                        sub_ds.set_epoch(epoch_idx, int(epochs))

            train_metrics = self.train_epoch(train_loader)
            if val_loader is not None and len(val_loader) > 0:
                val_metrics = self.validate_epoch(val_loader)
                score = float(val_metrics["val_loss"])
            else:
                val_metrics = {
                    "val_loss": train_metrics["train_loss"],
                    "val_emg_loss": train_metrics["train_emg_loss"],
                    "val_action_loss": train_metrics["train_action_loss"],
                    "val_action_acc": train_metrics["train_action_acc"],
                }
                score = float(train_metrics["train_loss"])

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            row: dict[str, float] = {
                "epoch": float(epoch_idx + 1),
                **train_metrics,
                **val_metrics,
                "lr": current_lr,
            }
            history.append(row)
            if len(train_loader) > 20:
                print(
                    f"[Epoch {epoch_idx + 1:2d}/{int(epochs)}] "
                    f"train_loss={train_metrics['train_loss']:.4f} "
                    f"train_acc={train_metrics['train_action_acc']:.4f} | "
                    f"val_loss={val_metrics['val_loss']:.4f} "
                    f"val_acc={val_metrics['val_action_acc']:.4f} "
                    f"lr={current_lr:.2e}",
                    flush=True,
                )

            if score <= best_val_loss:
                best_val_loss = score
                save_reactemg_checkpoint(
                    best_ckpt_path,
                    self.model,
                    args_dict=args_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch_idx + 1,
                    best_val_loss=best_val_loss,
                )

        if not best_ckpt_path.exists():
            save_reactemg_checkpoint(
                best_ckpt_path,
                self.model,
                args_dict=args_to_save,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=int(epochs),
                best_val_loss=best_val_loss,
            )

        self.save_history(history, out_dir / "history.csv")
        return history

    @staticmethod
    def save_history(history: Sequence[dict[str, float]], path: str | Path) -> Path:
        """Write epoch-level metrics history to CSV."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "epoch",
            "train_loss",
            "train_emg_loss",
            "train_action_loss",
            "train_action_acc",
            "val_loss",
            "val_emg_loss",
            "val_action_loss",
            "val_action_acc",
            "lr",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return out_path

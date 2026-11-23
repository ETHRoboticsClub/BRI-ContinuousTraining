"""WaveFormer GPU/CPU trainer, cosine warmup scheduler, evaluation metrics, and checkpoints."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CLASS_NAMES_6, load_raw_recording
from src.metrics import LookaheadAggregator, aggregate_sequence_metrics, evaluate_transition_accuracy
from src.model.model import Any2Any_Model, build_model_from_config
from src.model.waveformer import WaveFormer
from src.preprocess import Preprocess, preprocess_waveformer_window


def resolve_device(device_pref: str | None = None) -> torch.device:
    """Resolve PyTorch device cleanly with CUDA auto-detection and CPU fallback."""
    pref = (device_pref or "auto").lower()
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """Cosine decay scheduler with linear warmup matching ForeverBlue816/WaveFormer/util/lr_sched.py."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_ratio = float(min_lr_ratio)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps + 1)
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def save_waveformer_checkpoint(
    path: str | Path,
    model: nn.Module,
    args_dict: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    epoch: int = 0,
    best_score: float = 0.0,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save WaveFormer checkpoint with full architectural metadata and state dict."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = int(args_dict.get("num_classes", getattr(model, "num_classes", 6)))
    payload: dict[str, Any] = {
        "arch": "waveformer",
        "model_state_dict": model.state_dict(),
        "model_info": {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        "args_dict": dict(args_dict),
        "class_names": list(CLASS_NAMES_6[:num_classes]),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, out_path)
    return out_path


def load_waveformer_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load WaveFormer checkpoint from ``path``."""
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict):
        args_dict = dict(ckpt.get("args_dict", {}))
        if model is None:
            args_dict.setdefault("arch", str(ckpt.get("arch", "waveformer")))
            model = build_model_from_config(args_dict)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model_info" in ckpt and "model_state_dict" in ckpt["model_info"]:
            state_dict = ckpt["model_info"]["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        if model is None:
            model = WaveFormer()
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(dev)
    return model, ckpt if isinstance(ckpt, dict) else {}


def _compute_classification_metrics(
    logits_all: np.ndarray,
    targets_all: np.ndarray,
    num_classes: int = 6,
) -> dict[str, Any]:
    if len(targets_all) == 0:
        return {
            "val_acc": 0.0,
            "val_balanced_acc": 0.0,
            "val_macro_f1": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_auroc": 0.0,
            "per_class_acc": {},
            "confusion_matrix": [],
        }

    probs = torch.softmax(torch.from_numpy(logits_all).float(), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)

    acc = float(accuracy_score(targets_all, preds))
    bacc = float(balanced_accuracy_score(targets_all, preds))
    f1 = float(f1_score(targets_all, preds, average="macro", zero_division=0))
    prec = float(precision_score(targets_all, preds, average="macro", zero_division=0))
    rec = float(recall_score(targets_all, preds, average="macro", zero_division=0))

    labels = list(range(num_classes))
    cm = confusion_matrix(targets_all, preds, labels=labels)
    per_class_acc: dict[str, float] = {}
    for c in range(num_classes):
        name = CLASS_NAMES_6[c] if c < len(CLASS_NAMES_6) else str(c)
        row_sum = int(cm[c].sum())
        per_class_acc[name] = float(cm[c, c] / row_sum) if row_sum > 0 else 0.0

    try:
        present = np.unique(targets_all)
        if len(present) == num_classes:
            auroc = float(roc_auc_score(targets_all, probs, multi_class="ovr", average="macro"))
        else:
            auroc = 0.0
    except Exception:
        auroc = 0.0

    return {
        "val_acc": acc,
        "val_balanced_acc": bacc,
        "val_macro_f1": f1,
        "val_precision": prec,
        "val_recall": rec,
        "val_auroc": auroc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm.tolist(),
    }


class WaveFormerTrainer:
    """GPU/CPU trainer for WaveFormer sEMG gesture classification."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 0.01,
        warmup_epochs: int = 3,
        label_smoothing: float = 0.05,
        grad_clip: float = 1.0,
        class_weights: Sequence[float] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = device if isinstance(device, torch.device) else resolve_device(str(device))
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.min_lr = float(min_lr)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.label_smoothing = float(label_smoothing)
        self.grad_clip = float(grad_clip)

        cw_tensor = (
            torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
            if class_weights is not None
            else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=self.label_smoothing)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler: LambdaLR | None = None
        self._use_bf16 = bool(self.device.type == "cuda" and torch.cuda.is_bf16_supported())

    def _autocast_ctx(self):
        if self._use_bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        iterator = tqdm(train_loader, desc="Train", leave=False, disable=len(train_loader) <= 20)
        for x_batch, y_batch in iterator:
            x_batch = x_batch.to(device=self.device, dtype=torch.float32, non_blocking=True)
            y_batch = y_batch.to(device=self.device, dtype=torch.long, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_ctx():
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

            loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            bsz = int(y_batch.size(0))
            loss_sum += float(loss.detach().item()) * bsz
            preds = torch.argmax(logits.detach(), dim=-1)
            correct += int((preds == y_batch).sum().item())
            total += bsz

        return {
            "train_loss": loss_sum / max(1, total),
            "train_acc": correct / max(1, total),
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        loss_sum = 0.0
        total = 0
        all_logits: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device=self.device, dtype=torch.float32, non_blocking=True)
            y_batch = y_batch.to(device=self.device, dtype=torch.long, non_blocking=True)
            with self._autocast_ctx():
                logits = self.model(x_batch)
                loss = F.cross_entropy(logits, y_batch, reduction="sum")

            loss_sum += float(loss.detach().item())
            total += int(y_batch.size(0))
            all_logits.append(logits.detach().float().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

        logits_cat = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 6), dtype=np.float32)
        targets_cat = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=np.int64)
        metrics = _compute_classification_metrics(
            logits_cat,
            targets_cat,
            num_classes=int(getattr(self.model, "num_classes", 6)),
        )
        metrics["val_loss"] = loss_sum / max(1, total)
        return metrics

    @torch.no_grad()
    def evaluate_streaming(
        self,
        val_files: Sequence[str | Path],
        window_size: int | None = None,
        step_size: int = 20,
        scale_divisor: float = 128.0,
        hand_choice: str = "right",
        remove_dc: bool = True,
        max_files: int = 300,
    ) -> dict[str, Any]:
        """Simulate causal 10 Hz sliding-window online inference over held-out full recordings."""
        self.model.eval()
        win_len = int(window_size or getattr(self.model, "window_size", 200))
        chans = int(getattr(self.model, "channels", 8))
        n_cls = int(getattr(self.model, "num_classes", 6))

        eval_files = list(val_files)[:max_files] if max_files > 0 else list(val_files)
        all_logits: list[np.ndarray] = []
        all_targets: list[int] = []
        trial_correct = 0
        trial_total = 0

        for path in eval_files:
            raw_emg, gt = load_raw_recording(path, channels=chans, num_classes=n_cls)
            if len(raw_emg) < win_len or np.all(gt < 0):
                continue

            windows_batch: list[np.ndarray] = []
            step_targets: list[int] = []
            for end_idx in range(win_len, len(raw_emg) + 1, max(1, int(step_size))):
                raw_win = raw_emg[end_idx - win_len : end_idx]
                tail_len = min(90, win_len)
                gt_tail = gt[end_idx - tail_len : end_idx]
                valid_tail = gt_tail[(gt_tail >= 0) & (gt_tail < n_cls)]
                if valid_tail.size == 0:
                    continue
                target_lbl = int(np.bincount(valid_tail.astype(np.int64), minlength=n_cls).argmax())
                cond_tc = preprocess_waveformer_window(
                    raw_win,
                    scale_divisor=scale_divisor,
                    hand_choice=hand_choice,
                    remove_dc=remove_dc,
                    channels=chans,
                    return_nchw=False,
                )
                windows_batch.append(cond_tc.T)
                step_targets.append(target_lbl)

            if not windows_batch:
                continue

            x_t = torch.from_numpy(np.stack(windows_batch, axis=0)).unsqueeze(1).to(self.device, dtype=torch.float32)
            with self._autocast_ctx():
                logits_np = self.model(x_t).detach().float().cpu().numpy()

            all_logits.append(logits_np)
            all_targets.extend(step_targets)

            # Trial-level decision over active span (or all steps for Relax trial)
            true_trial_cls = int(np.max(gt[gt >= 0]))
            preds_np = np.argmax(logits_np, axis=-1)
            if true_trial_cls == 0:
                pred_trial = int(np.bincount(preds_np, minlength=n_cls).argmax())
            else:
                active_steps = [i for i, y in enumerate(step_targets) if y == true_trial_cls]
                if active_steps:
                    pred_trial = int(np.bincount(preds_np[active_steps], minlength=n_cls).argmax())
                else:
                    pred_trial = int(np.bincount(preds_np, minlength=n_cls).argmax())
            trial_correct += int(pred_trial == true_trial_cls)
            trial_total += 1

        if not all_logits:
            return {
                "streaming_step_acc": 0.0,
                "streaming_balanced_acc": 0.0,
                "streaming_macro_f1": 0.0,
                "streaming_trial_acc": 0.0,
                "num_trials": 0,
            }

        cat_logits = np.concatenate(all_logits, axis=0)
        cat_targets = np.asarray(all_targets, dtype=np.int64)
        cls_metrics = _compute_classification_metrics(cat_logits, cat_targets, num_classes=n_cls)
        return {
            "streaming_step_acc": cls_metrics["val_acc"],
            "streaming_balanced_acc": cls_metrics["val_balanced_acc"],
            "streaming_macro_f1": cls_metrics["val_macro_f1"],
            "streaming_auroc": cls_metrics["val_auroc"],
            "streaming_per_class_acc": cls_metrics["per_class_acc"],
            "streaming_confusion_matrix": cls_metrics["confusion_matrix"],
            "streaming_trial_acc": float(trial_correct / max(1, trial_total)),
            "num_trials": trial_total,
            "num_steps": len(cat_targets),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 25,
        output_dir: str | Path = "output",
        checkpoint_args: dict[str, Any] | None = None,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, float]]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = out_dir / "best_model.pt"

        steps_per_epoch = max(1, len(train_loader))
        total_steps = max(1, int(epochs) * steps_per_epoch)
        warmup_steps = max(0, int(round(self.warmup_epochs * steps_per_epoch)))
        min_lr_ratio = max(1e-5, self.min_lr / max(1e-9, self.lr))
        self.scheduler = build_cosine_warmup_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

        start_epoch = 0
        best_score = -1.0
        if resume_from is not None and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            model_info = ckpt.get("model_info", {})
            state_dict = ckpt.get("model_state_dict") or model_info.get("model_state_dict")
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            if model_info.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(model_info["optimizer_state_dict"])
            if model_info.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(model_info["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_score = float(ckpt.get("best_score", -1.0))

        args_to_save = dict(checkpoint_args or {})
        args_to_save.setdefault("arch", "waveformer")
        args_to_save.setdefault("channels", getattr(self.model, "channels", 8))
        args_to_save.setdefault("num_classes", getattr(self.model, "num_classes", 6))
        args_to_save.setdefault("window_size", getattr(self.model, "window_size", 200))
        args_to_save.setdefault("patch_size", list(getattr(self.model, "patch_size", (1, 20))))
        args_to_save.setdefault("embed_dim", getattr(self.model, "embed_dim", 256))
        args_to_save.setdefault("depth", getattr(self.model, "depth", 6))
        args_to_save.setdefault("num_heads", getattr(self.model, "num_heads", 8))
        args_to_save.setdefault("mlp_ratio", getattr(self.model, "mlp_ratio", 2.0))
        args_to_save.setdefault("wt_levels", getattr(self.model, "wt_levels", 3))
        args_to_save.setdefault("wt_type", getattr(self.model, "wt_type", "db1"))
        args_to_save.setdefault("wt_kernel_size", getattr(self.model, "wt_kernel_size", 3))
        args_to_save.setdefault("highfreq_dropout", getattr(self.model, "highfreq_dropout", 0.1))
        args_to_save.setdefault("dropout", getattr(self.model, "dropout", 0.1))
        args_to_save.setdefault("drop_path", getattr(self.model, "drop_path", 0.1))

        history: list[dict[str, float]] = []
        for epoch_idx in range(start_epoch, int(epochs)):
            train_metrics = self.train_epoch(train_loader)
            if val_loader is not None and len(val_loader) > 0:
                val_full = self.validate_epoch(val_loader)
                score = float(val_full["val_macro_f1"])
            else:
                val_full = {
                    "val_loss": train_metrics["train_loss"],
                    "val_acc": train_metrics["train_acc"],
                    "val_balanced_acc": train_metrics["train_acc"],
                    "val_macro_f1": train_metrics["train_acc"],
                    "val_auroc": 0.0,
                    "per_class_acc": {},
                }
                score = float(train_metrics["train_acc"])

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            row: dict[str, float] = {
                "epoch": float(epoch_idx + 1),
                "train_loss": float(train_metrics["train_loss"]),
                "train_acc": float(train_metrics["train_acc"]),
                "val_loss": float(val_full["val_loss"]),
                "val_acc": float(val_full["val_acc"]),
                "val_balanced_acc": float(val_full["val_balanced_acc"]),
                "val_macro_f1": float(val_full["val_macro_f1"]),
                "val_auroc": float(val_full["val_auroc"]),
                "lr": current_lr,
            }
            history.append(row)

            per_cls_str = ""
            if isinstance(val_full.get("per_class_acc"), dict) and val_full["per_class_acc"]:
                per_cls_str = " | " + " ".join(f"{k}:{v * 100:.0f}%" for k, v in val_full["per_class_acc"].items())
            if len(train_loader) > 5 or epoch_idx == int(epochs) - 1:
                print(
                    f"[Epoch {epoch_idx + 1:2d}/{int(epochs)}] "
                    f"tr_loss={row['train_loss']:.4f} tr_acc={row['train_acc']:.4f} | "
                    f"va_loss={row['val_loss']:.4f} va_acc={row['val_acc']:.4f} "
                    f"va_bacc={row['val_balanced_acc']:.4f} va_f1={row['val_macro_f1']:.4f} "
                    f"lr={current_lr:.2e}{per_cls_str}",
                    flush=True,
                )

            if score >= best_score:
                best_score = score
                save_waveformer_checkpoint(
                    best_ckpt_path,
                    self.model,
                    args_dict=args_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch_idx + 1,
                    best_score=best_score,
                    metrics=val_full,
                )

        if not best_ckpt_path.exists():
            save_waveformer_checkpoint(
                best_ckpt_path,
                self.model,
                args_dict=args_to_save,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=int(epochs),
                best_score=best_score,
            )

        self.save_history(history, out_dir / "history.csv")
        return history

    @staticmethod
    def save_history(history: Sequence[dict[str, float]], path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_balanced_acc",
            "val_macro_f1",
            "val_auroc",
            "lr",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return out_path


# ---------------------------------------------------------------------------
# Legacy ReactEMG helpers kept for backward compatibility with older scripts
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    lr_decay_method: str = "linear",
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(round(total_steps * float(warmup_ratio))))
    decay_method = str(lr_decay_method).lower()

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps + 1)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
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
    (
        emg_window,
        action_window,
        _masked_emg,
        masked_actions,
        mask_positions_emg,
        mask_positions_actions,
        task_idx,
        _transition_idx,
    ) = batch
    emg_window = emg_window.to(device=device, dtype=torch.float32)
    action_window = action_window.to(device=device, dtype=torch.long)
    masked_actions = masked_actions.to(device=device, dtype=torch.long)
    mask_positions_emg = mask_positions_emg.to(device=device, dtype=torch.bool)
    mask_positions_actions = mask_positions_actions.to(device=device, dtype=torch.bool)
    task_idx = task_idx.to(device=device, dtype=torch.long)

    emg_out, action_out = model(emg_window, masked_actions, task_idx, mask_positions_emg)
    emg_count = int(mask_positions_emg.sum().item())
    emg_loss = (
        F.mse_loss(emg_out[mask_positions_emg], emg_window[mask_positions_emg], reduction="mean")
        if emg_count > 0
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    valid_action_mask = mask_positions_actions & (action_window >= 0) & (action_window < int(num_classes))
    action_count = int(valid_action_mask.sum().item())
    if action_count > 0:
        masked_logits = action_out[valid_action_mask]
        masked_targets = action_window[valid_action_mask]
        action_loss = F.cross_entropy(masked_logits, masked_targets, reduction="mean")
        action_correct = int(
            (torch.argmax(masked_logits[:, : int(num_classes)], dim=-1) == masked_targets).sum().item()
        )
    else:
        action_loss = torch.zeros((), device=device, dtype=torch.float32)
        action_correct = 0

    emg_weight = 100.0 if float(scale_emg_loss) == 1.0 else max(1.0, float(scale_emg_loss))
    if emg_count > 0 and action_count > 0:
        total_loss = emg_weight * emg_loss + action_loss
    elif emg_count > 0:
        total_loss = emg_weight * emg_loss
    elif action_count > 0:
        total_loss = action_loss
    else:
        total_loss = 0.0 * (emg_out.sum() + action_out.sum())

    return {
        "loss": total_loss,
        "emg_loss": float(emg_loss.detach().item()),
        "action_loss": float(action_loss.detach().item()),
        "emg_count": emg_count,
        "action_count": action_count,
        "action_correct": action_correct,
        "total_mask_positions": int(mask_positions_emg.sum().item() + mask_positions_actions.sum().item()),
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
    """Legacy ReactEMG trainer retained for compatibility."""

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
        return torch.amp.autocast("cuda", dtype=torch.bfloat16) if self._use_bf16 else nullcontext()

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss_sum = 0.0
        total_weight_sum = 0
        act_correct_total = 0
        act_count_total = 0
        for batch in train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_ctx():
                res = process_batch(
                    self.model,
                    batch,
                    device=self.device,
                    scale_emg_loss=self.scale_emg_loss,
                    num_classes=self.model.num_classes,
                )
            res["loss"].backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            w = res["total_mask_positions"] if res["total_mask_positions"] > 0 else int(batch[0].size(0))
            total_loss_sum += float(res["loss"].detach().item()) * w
            total_weight_sum += w
            act_count_total += res["action_count"]
            act_correct_total += res["action_correct"]
        return {
            "train_loss": total_loss_sum / max(1, total_weight_sum),
            "train_emg_loss": 0.0,
            "train_action_loss": 0.0,
            "train_action_acc": act_correct_total / max(1, act_count_total),
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss_sum = 0.0
        total_weight_sum = 0
        act_correct_total = 0
        act_count_total = 0
        for batch in val_loader:
            with self._autocast_ctx():
                res = process_batch(
                    self.model,
                    batch,
                    device=self.device,
                    scale_emg_loss=self.scale_emg_loss,
                    num_classes=self.model.num_classes,
                )
            w = res["total_mask_positions"] if res["total_mask_positions"] > 0 else int(batch[0].size(0))
            total_loss_sum += float(res["loss"].detach().item()) * w
            total_weight_sum += w
            act_count_total += res["action_count"]
            act_correct_total += res["action_correct"]
        return {
            "val_loss": total_loss_sum / max(1, total_weight_sum),
            "val_emg_loss": 0.0,
            "val_action_loss": 0.0,
            "val_action_acc": act_correct_total / max(1, act_count_total),
        }

    @torch.no_grad()
    def evaluate_streaming(self, val_files: Sequence[str | Path], preprocess: Preprocess | None = None, **kwargs: Any):
        aggregator = LookaheadAggregator(
            window_size=self.model.window_size,
            lookahead=int(kwargs.get("lookahead", 50)),
            samples_between_prediction=int(kwargs.get("samples_between_prediction", 20)),
            num_classes=self.model.num_classes,
        )
        per_file: list[dict[str, object]] = []
        for path in val_files:
            emg_tc, gt = load_raw_recording(path, channels=self.model.channels, num_classes=self.model.num_classes)
            prep = preprocess if preprocess is not None else Preprocess(channels=self.model.channels, mode="reactemg")
            emg_tc = prep.preprocess_sequence(emg_tc)
            if len(emg_tc) < self.model.window_size or np.all(gt < 0):
                continue
            preds, _ = aggregator.predict_sequence(
                self.model, emg_tc, stride=int(kwargs.get("stride", 1)), device=self.device, pad_initial=True
            )
            per_file.append(evaluate_transition_accuracy(preds, gt, buffer_range=int(kwargs.get("buffer_range", 200))))
        return aggregate_sequence_metrics(per_file)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 12,
        output_dir: str | Path = "output",
        checkpoint_args: dict[str, Any] | None = None,
        resume_from: str | Path | None = None,
    ):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = out_dir / "best_model.pt"
        self.scheduler = build_scheduler(
            self.optimizer, max(1, epochs * max(1, len(train_loader))), self.warmup_ratio, self.lr_decay_method
        )
        start_epoch = 0
        best_val_loss = float("inf")
        if resume_from is not None and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            mi = ckpt.get("model_info", {})
            if "model_state_dict" in mi:
                self.model.load_state_dict(mi["model_state_dict"])
            if mi.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(mi["optimizer_state_dict"])
            if mi.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(mi["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        history = []
        for epoch_idx in range(start_epoch, int(epochs)):
            tr = self.train_epoch(train_loader)
            va = (
                self.validate_epoch(val_loader)
                if val_loader is not None
                else {
                    "val_loss": tr["train_loss"],
                    "val_emg_loss": 0.0,
                    "val_action_loss": 0.0,
                    "val_action_acc": tr["train_action_acc"],
                }
            )
            row = {"epoch": float(epoch_idx + 1), **tr, **va, "lr": float(self.optimizer.param_groups[0]["lr"])}
            history.append(row)
            if va["val_loss"] <= best_val_loss:
                best_val_loss = va["val_loss"]
                save_reactemg_checkpoint(
                    best_ckpt_path,
                    self.model,
                    dict(checkpoint_args or {}),
                    self.optimizer,
                    self.scheduler,
                    epoch_idx + 1,
                    best_val_loss,
                )
        if not best_ckpt_path.exists():
            save_reactemg_checkpoint(
                best_ckpt_path,
                self.model,
                dict(checkpoint_args or {}),
                self.optimizer,
                self.scheduler,
                int(epochs),
                best_val_loss,
            )
        WaveFormerTrainer.save_history(history, out_dir / "history.csv")
        return history

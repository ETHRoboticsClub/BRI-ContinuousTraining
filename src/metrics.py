"""ReactEMG streaming look-ahead aggregation and transition/maintenance evaluation metrics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-12)


def aggregate_logits(
    window_logits: np.ndarray,
    weighting_method: str = "equal",
    weight_max_factor: float = 1.5,
    likelihood_format: str = "probs",
) -> tuple[int, np.ndarray]:
    """Aggregate overlapping per-window logits ``(N, C)`` into one decision and distribution.

    Args:
        window_logits: Array of shape ``(N, C)`` sorted oldest-window to newest-window.
        weighting_method: ``"equal"`` or ``"weighted"`` (linear ramp toward newest window).
        weight_max_factor: Maximum ramp weight multiplier for the newest window.
        likelihood_format: ``"probs"`` (apply softmax before averaging) or ``"logits"``.

    Returns:
        Tuple ``(pred_class, aggregated_probs)`` where ``aggregated_probs`` sums to 1.
    """
    arr = np.asarray(window_logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] == 0:
        raise ValueError("Cannot aggregate empty window_logits")

    if likelihood_format == "probs":
        values = _softmax(arr, axis=-1)
    elif likelihood_format == "logits":
        values = arr
    else:
        raise ValueError(f"Unsupported likelihood_format {likelihood_format!r}")

    n = values.shape[0]
    if weighting_method == "equal" or n == 1:
        weights = np.full((n, 1), 1.0 / n, dtype=np.float64)
    elif weighting_method in ("weighted", "linear"):
        ramp = np.linspace(1.0, float(weight_max_factor), n, dtype=np.float64)
        weights = (ramp / np.sum(ramp))[:, np.newaxis]
    else:
        raise ValueError(f"Unsupported weighting_method {weighting_method!r}")

    agg = np.sum(values * weights, axis=0)
    if likelihood_format == "logits":
        probs = _softmax(agg, axis=-1).astype(np.float32)
    else:
        total = float(np.sum(agg))
        probs = (agg / max(total, 1e-12)).astype(np.float32)

    pred_class = int(np.argmax(probs))
    return pred_class, probs


@dataclass
class LookaheadAggregator:
    """Online and offline smoothed look-ahead aggregator from ReactEMG Section III-C.

    At sample rate ``fs=200 Hz``, ``lookahead=50`` collects overlapping window
    predictions covering target timestep ``tau`` over ``[tau, tau + 50]`` (0.25 s)
    and updates the emitted gesture every ``samples_between_prediction=20`` (10 Hz).
    """

    window_size: int = 600
    lookahead: int = 50
    samples_between_prediction: int = 20
    maj_vote_range: str = "future"
    likelihood_format: str = "probs"
    weight_max_factor: float = 1.5
    weighting_method: str = "equal"
    num_classes: int = 6

    _current_step: int = field(default=0, init=False, repr=False)
    _last_decision_step: int = field(default=-10_000_000, init=False, repr=False)
    _last_pred: int = field(default=0, init=False, repr=False)
    _last_probs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32), init=False, repr=False)
    # Maps global timestep tau -> deque of logits rows from windows ending at tau, tau+1, ..., tau+lookahead
    _timestep_logits: dict[int, list[np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _completed_history: deque[int] = field(default_factory=lambda: deque(maxlen=4096), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lookahead < 0 or self.lookahead >= self.window_size:
            raise ValueError(
                f"lookahead ({self.lookahead}) must satisfy 0 <= lookahead < window_size ({self.window_size})"
            )
        if self.samples_between_prediction < 1:
            raise ValueError(f"samples_between_prediction must be >= 1, got {self.samples_between_prediction}")
        if self.maj_vote_range not in ("future", "single", "all"):
            raise ValueError(f"Unsupported maj_vote_range {self.maj_vote_range!r}")
        self.reset()

    def reset(self) -> None:
        self._current_step = 0
        self._last_decision_step = -10_000_000
        self._last_pred = 0
        self._last_probs = np.full(self.num_classes, 1.0 / max(1, self.num_classes), dtype=np.float32)
        self._timestep_logits.clear()
        self._completed_history.clear()

    def push_window_logits(self, logits_wc: np.ndarray, step_increment: int = 1) -> tuple[int, np.ndarray]:
        """Push logits ``(W, num_classes)`` from window ending at current streaming step.

        Args:
            logits_wc: Logits matrix of shape ``(window_size, num_classes)`` for the
                most recent window ending at ``current_step``.
            step_increment: Sample stride represented by this update (>= 1).

        Returns:
            Tuple ``(held_pred_class, held_class_probs)`` representing the smoothed
            look-ahead classification decision held for up to ``samples_between_prediction``.
        """
        arr = np.asarray(logits_wc, dtype=np.float32)
        if arr.shape != (self.window_size, self.num_classes):
            raise ValueError(f"Expected logits shape ({self.window_size}, {self.num_classes}), got {arr.shape}")

        end_t = self._current_step
        start_t = end_t - self.window_size + 1
        # Store logit estimates for recent timesteps needed within the lookahead horizon
        horizon_start = max(start_t, end_t - self.lookahead - self.samples_between_prediction)
        for tau in range(horizon_start, end_t + 1):
            local_idx = tau - start_t
            self._timestep_logits.setdefault(tau, []).append(arr[local_idx])

        target_tau = max(start_t, end_t - self.lookahead)
        should_update = (end_t - self._last_decision_step) >= self.samples_between_prediction or (
            self._last_decision_step < 0
        )

        if should_update and target_tau in self._timestep_logits:
            candidates = self._timestep_logits[target_tau]
            if self.maj_vote_range == "single":
                candidates = [candidates[-1]]
            stacked = np.stack(candidates, axis=0)
            pred, probs = aggregate_logits(
                stacked,
                weighting_method=self.weighting_method,
                weight_max_factor=self.weight_max_factor,
                likelihood_format=self.likelihood_format,
            )
            self._last_pred = pred
            self._last_probs = probs
            self._last_decision_step = end_t
            self._completed_history.append(pred)

        # Prune entries older than target_tau - samples_between_prediction
        prune_before = target_tau - self.samples_between_prediction
        for stale_t in [k for k in self._timestep_logits if k < prune_before]:
            del self._timestep_logits[stale_t]

        self._current_step += max(1, int(step_increment))
        return self._last_pred, self._last_probs.copy()

    @torch.no_grad()
    def predict_sequence(
        self,
        model: torch.nn.Module,
        emg_tc: np.ndarray,
        stride: int = 1,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``model`` over a full sequence ``(T, C)`` with smoothed look-ahead.

        Returns:
            Tuple ``(preds_t, probs_tc)`` aligned to timesteps ``0..T-1``.
            Positions before the first full window carry ``-1`` and uniform probs,
            while covered timesteps hold the look-ahead decision with hold cadence
            ``samples_between_prediction``.
        """
        seq = np.asarray(emg_tc, dtype=np.float32)
        t_len, channels = seq.shape
        preds = np.full(t_len, -1, dtype=np.int64)
        probs_seq = np.full((t_len, self.num_classes), 1.0 / self.num_classes, dtype=np.float32)
        if t_len < self.window_size:
            return preds, probs_seq

        eval_stride = max(1, int(stride))
        starts = list(range(0, t_len - self.window_size + 1, eval_stride))
        if not starts:
            return preds, probs_seq

        model.eval()
        dev = torch.device(device)
        all_logits: list[np.ndarray] = []

        for b_start in range(0, len(starts), max(1, int(batch_size))):
            chunk_starts = starts[b_start : b_start + max(1, int(batch_size))]
            windows = np.stack([seq[s : s + self.window_size] for s in chunk_starts], axis=0)
            emg_tensor = torch.from_numpy(windows).to(dev)
            if hasattr(model, "predict_action_logits") and callable(model.predict_action_logits):
                class_logits_t = model.predict_action_logits(emg_tensor)
            else:
                mask_actions = torch.full(
                    (len(chunk_starts), self.window_size), self.num_classes, dtype=torch.long, device=dev
                )
                task_idx = torch.zeros(len(chunk_starts), dtype=torch.long, device=dev)
                mask_pos_emg = torch.zeros_like(emg_tensor, dtype=torch.bool)
                _, act_logits = model(emg_tensor, mask_actions, task_idx, mask_pos_emg)
                class_logits_t = act_logits[:, :, : self.num_classes]
            class_logits = class_logits_t.detach().cpu().numpy()
            all_logits.append(class_logits)

        window_logits = np.concatenate(all_logits, axis=0)  # (N_win, W, num_classes)

        # Accumulate logits per timestep tau covered by each window ending at e = s + W - 1
        per_t_buckets: list[list[np.ndarray]] = [[] for _ in range(t_len)]
        per_t_single: list[np.ndarray | None] = [None] * t_len
        for idx, s in enumerate(starts):
            end_t = s + self.window_size - 1
            for tau in range(s, end_t + 1):
                # Only use windows whose end index does not exceed tau + lookahead
                if end_t <= tau + self.lookahead:
                    local_pos = tau - s
                    logit_vec = window_logits[idx, local_pos]
                    per_t_buckets[tau].append(logit_vec)
                    if end_t == tau + self.lookahead or (end_t <= tau + self.lookahead and per_t_single[tau] is None):
                        per_t_single[tau] = logit_vec

        # Determine predicted decision at timesteps from first covered sample to end
        first_covered = starts[0] + self.window_size - 1
        decision_pred: int | None = None
        decision_probs: np.ndarray | None = None
        next_update_t = first_covered

        for cur_t in range(first_covered, t_len):
            target_tau = max(0, cur_t - self.lookahead)
            if cur_t >= next_update_t or decision_pred is None:
                bucket = per_t_buckets[target_tau]
                if self.maj_vote_range == "single" and per_t_single[target_tau] is not None:
                    bucket = [per_t_single[target_tau]]  # type: ignore[list-item]
                if bucket:
                    stacked = np.stack(bucket, axis=0)
                    decision_pred, decision_probs = aggregate_logits(
                        stacked,
                        weighting_method=self.weighting_method,
                        weight_max_factor=self.weight_max_factor,
                        likelihood_format=self.likelihood_format,
                    )
                    next_update_t = cur_t + self.samples_between_prediction
            if decision_pred is not None and decision_probs is not None:
                preds[cur_t] = decision_pred
                probs_seq[cur_t] = decision_probs

        return preds, probs_seq


def extract_buffer_windows(
    gt: np.ndarray,
    buffer_range: int = 200,
) -> list[dict[str, int]]:
    """Extract reaction buffer spans centered on ground-truth intent transitions.

    Each transition ``tau`` where ``gt[tau] != gt[tau - 1]`` yields a buffer
    ``[start, end)`` of width ``buffer_range`` (default ``+-100`` samples = ``+-0.5 s``).
    """
    actions = np.asarray(gt, dtype=np.int64).reshape(-1)
    t_len = len(actions)
    if t_len < 2:
        return []

    half = max(1, int(buffer_range) // 2)
    trans_indices = np.where((actions[1:] != actions[:-1]) & (actions[1:] >= 0) & (actions[:-1] >= 0))[0] + 1
    windows: list[dict[str, int]] = []
    for tau in trans_indices:
        tau_i = int(tau)
        windows.append(
            {
                "transition_idx": tau_i,
                "from_class": int(actions[tau_i - 1]),
                "to_class": int(actions[tau_i]),
                "start": max(0, tau_i - half),
                "end": min(t_len, tau_i + half),
            }
        )
    return windows


def extract_maintenance_windows(
    gt: np.ndarray,
    buffer_range: int = 200,
) -> list[dict[str, int]]:
    """Extract steady-hold maintenance intervals strictly between transition buffers."""
    actions = np.asarray(gt, dtype=np.int64).reshape(-1)
    t_len = len(actions)
    if t_len == 0:
        return []

    half = max(1, int(buffer_range) // 2)
    buffers = extract_buffer_windows(actions, buffer_range=buffer_range)
    if not buffers:
        valid = np.where(actions >= 0)[0]
        if len(valid) == 0:
            return []
        return [{"start": int(valid[0]), "end": int(valid[-1] + 1), "target_class": int(actions[valid[0]])}]

    intervals: list[dict[str, int]] = []
    boundaries: list[tuple[int, int]] = [(0, buffers[0]["start"])]
    for i in range(len(buffers) - 1):
        boundaries.append((buffers[i]["end"], buffers[i + 1]["start"]))
    boundaries.append((buffers[-1]["end"], t_len))

    for start, end in boundaries:
        if end > start:
            cls_val = int(actions[start])
            if cls_val >= 0:
                intervals.append({"start": int(start), "end": int(end), "target_class": cls_val})

    _ = half
    return intervals


def evaluate_transition_accuracy(
    preds: np.ndarray,
    gt: np.ndarray,
    buffer_range: int = 200,
    fs: int = 200,
    allow_relax: int = 0,
    strict_transition: int = 0,
) -> dict[str, object]:
    """Compute ReactEMG Raw Accuracy, Transition Accuracy, and Prediction Offset (ms).

    An event (transition + following maintenance hold) passes if:
    1. Inside the reaction buffer around the transition, the predicted sequence switches to
       ``to_class`` (without spurious extra wrong classes when ``strict_transition`` > 0), and
    2. Throughout the subsequent maintenance segment outside the buffers, predictions remain
       on ``target_class`` (optionally allowing brief ``Relax=0`` when ``allow_relax`` > 0).
    """
    pred_arr = np.asarray(preds, dtype=np.int64).reshape(-1)
    gt_arr = np.asarray(gt, dtype=np.int64).reshape(-1)
    if len(pred_arr) != len(gt_arr):
        raise ValueError(f"Length mismatch: preds={len(pred_arr)} vs gt={len(gt_arr)}")

    valid_mask = (gt_arr >= 0) & (pred_arr >= 0)
    if np.any(valid_mask):
        raw_accuracy = float(np.mean(pred_arr[valid_mask] == gt_arr[valid_mask]))
    else:
        raw_accuracy = 0.0

    buffers = extract_buffer_windows(gt_arr, buffer_range=buffer_range)
    reasons: dict[str, int] = {
        "missed_transition": 0,
        "wrong_class_in_buffer": 0,
        "maintenance_flicker": 0,
    }
    offsets_ms: list[float] = []

    half = max(1, int(buffer_range) // 2)
    passed = 0
    total = 0

    for idx, buf in enumerate(buffers):
        s, e = buf["start"], buf["end"]
        tau = buf["transition_idx"]
        from_cls = buf["from_class"]
        to_cls = buf["to_class"]

        buf_preds = pred_arr[s:e]
        # Skip events completely outside covered online predictions (e.g. warmup period < window_size)
        if np.all(buf_preds < 0):
            continue

        total += 1
        hit_rel = np.where(buf_preds == to_cls)[0]
        if len(hit_rel) == 0:
            reasons["missed_transition"] += 1
            continue

        first_hit_global = s + int(hit_rel[0])
        offsets_ms.append((first_hit_global - tau) * (1000.0 / float(fs)))

        if strict_transition:
            allowed_in_buf = {from_cls, to_cls}
            if allow_relax:
                allowed_in_buf.add(0)
            valid_buf = buf_preds[buf_preds >= 0]
            if any(int(p) not in allowed_in_buf for p in valid_buf):
                reasons["wrong_class_in_buffer"] += 1
                continue

        # Maintenance hold segment right after this transition until the next buffer
        next_start = buffers[idx + 1]["start"] if idx + 1 < len(buffers) else len(gt_arr)
        maint_start = min(e, next_start)
        maint_end = next_start
        if maint_end > maint_start:
            maint_preds = pred_arr[maint_start:maint_end]
            valid_maint = maint_preds[maint_preds >= 0]
            allowed_maint = {to_cls, 0} if allow_relax else {to_cls}
            if len(valid_maint) > 0 and any(int(p) not in allowed_maint for p in valid_maint):
                reasons["maintenance_flicker"] += 1
                continue

        passed += 1

    _ = half
    failed = total - passed
    transition_accuracy = float(passed / total) if total > 0 else raw_accuracy
    mean_offset_ms = float(np.mean(offsets_ms)) if offsets_ms else 0.0
    median_offset_ms = float(np.median(offsets_ms)) if offsets_ms else 0.0

    return {
        "raw_accuracy": raw_accuracy,
        "transition_accuracy": transition_accuracy,
        "event_counts": {"total": total, "passed": passed, "failed": failed},
        "transition_reasons": reasons,
        "offsets_ms": offsets_ms,
        "mean_offset_ms": mean_offset_ms,
        "median_offset_ms": median_offset_ms,
    }


def aggregate_sequence_metrics(results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Combine per-recording evaluation dicts into overall validation metrics."""
    if not results:
        return {
            "raw_accuracy": 0.0,
            "transition_accuracy": 0.0,
            "event_counts": {"total": 0, "passed": 0, "failed": 0},
            "transition_reasons": {
                "missed_transition": 0,
                "wrong_class_in_buffer": 0,
                "maintenance_flicker": 0,
            },
            "mean_offset_ms": 0.0,
            "median_offset_ms": 0.0,
        }

    raw_accs = [float(r["raw_accuracy"]) for r in results]
    total_events = sum(int(r["event_counts"]["total"]) for r in results)  # type: ignore[index]
    passed_events = sum(int(r["event_counts"]["passed"]) for r in results)  # type: ignore[index]
    failed_events = total_events - passed_events

    reasons = {"missed_transition": 0, "wrong_class_in_buffer": 0, "maintenance_flicker": 0}
    all_offsets: list[float] = []
    for r in results:
        r_reasons = r["transition_reasons"]  # type: ignore[index]
        for k in reasons:
            reasons[k] += int(r_reasons.get(k, 0))  # type: ignore[union-attr]
        all_offsets.extend(float(x) for x in r.get("offsets_ms", []))  # type: ignore[arg-type]

    trans_acc = (
        float(passed_events / total_events)
        if total_events > 0
        else float(np.mean([float(r["transition_accuracy"]) for r in results]))
    )

    return {
        "raw_accuracy": float(np.mean(raw_accs)),
        "transition_accuracy": trans_acc,
        "event_counts": {"total": total_events, "passed": passed_events, "failed": failed_events},
        "transition_reasons": reasons,
        "mean_offset_ms": float(np.mean(all_offsets)) if all_offsets else 0.0,
        "median_offset_ms": float(np.median(all_offsets)) if all_offsets else 0.0,
    }

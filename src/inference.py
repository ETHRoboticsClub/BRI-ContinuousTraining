"""Real-Time and Offline Inference Engine for Temporal Riemannian Regressor (TRR).

This module implements the inference pipeline for the TRR model based on Script 2 (infer_trr.py)
from Colot et al. (2026), supporting both real-time streaming FIFO queue inference and
high-throughput batch recording decoding.

Sections:
1. Constants & Unit Conversions: Default sampling rate (200.0 Hz), window timings, and rad/deg math.
2. Streaming Inference Engine (StreamingTRRInference): Stateful rolling buffer and 10-step FIFO queue.
3. Batch Recording Decoding (infer_recording): Vectorized offline prediction over full recordings.
4. Model & Artifact Loader (load_inference_artifacts): Loads weights, params.yaml, and metadata.
"""

import time
from collections import deque
from math import gcd, pi
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from scipy import signal

from src.model import Model
from src.preprocessing import preprocess

# ==============================================================================
# Section 1: Constants & Unit Conversions
# ==============================================================================

DEFAULT_SAMPLING_RATE_HZ: float = 200.0
MODEL_OPERATING_RATE_HZ: float = 200.0
SUB_WINDOW_DURATION_SEC: float = 0.300  # 300 ms sub-window
STEP_DURATION_SEC: float = 0.100  # 100 ms step interval (10 Hz update)
TOTAL_WINDOW_DURATION_SEC: float = 1.200  # 1200 ms context window
DEFAULT_SEQUENCE_LENGTH: int = 10  # 10 sub-windows
DEG_TO_RAD: float = pi / 180.0
RAD_TO_DEG: float = 180.0 / pi


def convert_angles(angles: np.ndarray, from_unit: str = "deg", to_unit: str = "deg") -> np.ndarray:
    """Convert joint angle predictions between degrees and radians.

    Args:
        angles: Array of joint angle predictions.
        from_unit: Source unit ('deg' or 'rad').
        to_unit: Target unit ('deg' or 'rad').

    Returns:
        Converted joint angles.
    """
    if from_unit == to_unit:
        return angles
    if from_unit == "deg" and to_unit == "rad":
        return angles * DEG_TO_RAD
    if from_unit == "rad" and to_unit == "deg":
        return angles * RAD_TO_DEG
    raise ValueError(f"Unsupported unit conversion from {from_unit} to {to_unit}")


# ==============================================================================
# Section 2: Model & Artifact Loader
# ==============================================================================


def load_inference_artifacts(
    checkpoint_path: Union[str, Path] = "runs/cross_subject/best_model.pt",
    config_path: Union[str, Path] = "params.yaml",
    device: str = "auto",
) -> Tuple[Model, Dict[str, Any], torch.device]:
    """Load trained TRR model weights and configuration.

    Args:
        checkpoint_path: Path to PyTorch model weights (.pt).
        config_path: Path to params.yaml.
        device: Target execution device ('auto', 'cuda', 'cpu').

    Returns:
        Tuple of (model, config, resolved_device).
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")

    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model_cfg = cfg.get("model", {})
    # Default to 8 channels single-band, 15 target joints
    in_channels = 8 if not cfg.get("preprocessing", {}).get("multiband", False) else 24
    out_dim = 15

    model = Model(in_channels=in_channels, out_dim=out_dim, **model_cfg)
    state_dict = torch.load(checkpoint_file, map_location=target_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    return model, cfg, target_device


# ==============================================================================
# Section 3: Stateful Streaming Inference Engine
# ==============================================================================


class StreamingTRRInference:
    """Stateful real-time streaming inference engine for TRR models.

    Accomplishes:
    1. Ingests incremental sEMG buffers (e.g. 100 ms chunks at user-supplied fs).
    2. Maintains a 300 ms rolling raw EMG buffer (60 samples at 200 Hz).
    3. Extracts OAS regularized covariance and Riemannian tangent space features (36-D or 108-D).
    4. Maintains a 10-step FIFO feature queue spanning 1200 ms of context.
    5. Executes forward pass through TRR recurrent network with latency profiling.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "runs/cross_subject/best_model.pt",
        config_path: Union[str, Path] = "params.yaml",
        fs: float = DEFAULT_SAMPLING_RATE_HZ,
        device: str = "auto",
        units: str = "deg",
        standardize_input: bool = True,
    ) -> None:
        """Initialize the streaming TRR inference engine.

        Args:
            checkpoint_path: Path to saved model weights.
            config_path: Path to params.yaml.
            fs: Input stream sampling frequency in Hz (default: 200.0).
            device: 'auto', 'cuda', or 'cpu'.
            units: Output unit ('deg' or 'rad'). Defaults to 'deg'.
            standardize_input: Whether to apply per-channel zero-mean unit-variance scaling.
        """
        self.fs: float = float(fs)
        self.model_fs: float = MODEL_OPERATING_RATE_HZ
        self.units: str = units.lower()
        self.standardize_input: bool = standardize_input

        # Load model and config
        self.model, self.cfg, self.device = load_inference_artifacts(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )

        # Buffer dimensions
        self.sub_window_len: int = int(round(SUB_WINDOW_DURATION_SEC * self.model_fs))  # 60 samples @ 200 Hz
        self.step_len: int = int(round(STEP_DURATION_SEC * self.model_fs))  # 20 samples @ 200 Hz
        self.input_step_len: int = int(round(STEP_DURATION_SEC * self.fs))  # samples per input chunk
        self.seq_len: int = DEFAULT_SEQUENCE_LENGTH  # 10 sub-windows
        self.in_channels: int = self.model.rfe.in_channels

        # Per-channel standardization stats
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # Internal stateful buffers
        self.raw_buffer: np.ndarray = np.zeros((self.in_channels, self.sub_window_len), dtype=np.float32)
        self.feature_queue: deque = deque(maxlen=self.seq_len)
        self.warmup_steps: int = 0

    def reset(self) -> None:
        """Reset internal rolling buffers and feature queue for a new session/recording."""
        self.raw_buffer.fill(0.0)
        self.feature_queue.clear()
        self.warmup_steps = 0

    def set_standardization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set channel standardization statistics for cross-subject mode.

        Args:
            mean: Channel mean array of shape (channels, 1) or (channels,).
            std: Channel standard deviation array of shape (channels, 1) or (channels,).
        """
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1) + 1e-6

    def _extract_single_cmts(self, raw_subwindow: torch.Tensor) -> torch.Tensor:
        """Extract multi-band Riemannian CMTS feature vector from a single 300 ms sub-window.

        Args:
            raw_subwindow: Tensor of shape (1, in_channels, sub_window_len).

        Returns:
            Projected tangent feature vector of shape (1, total_feat_dim).
        """
        band_feats: List[torch.Tensor] = []
        rfe = self.model.rfe
        for b in range(rfe.n_bands):
            xb = raw_subwindow[:, b * rfe.ch_per_band : (b + 1) * rfe.ch_per_band, :]
            cov = rfe._oas_cov(xb)
            tf = rfe._tangent_space(cov)
            band_feats.append(tf)
        return torch.cat(band_feats, dim=-1)  # [1, total_feat_dim]

    @torch.no_grad()
    def step(self, emg_chunk: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process one incoming sEMG buffer chunk and predict 15 joint angles.

        Args:
            emg_chunk: Incremental EMG chunk of shape (channels, n_samples) or (n_samples, channels).

        Returns:
            Tuple of (predicted_angles_15, step_latency_ms).
        """
        t_start = time.perf_counter()
        chunk = np.asarray(emg_chunk, dtype=np.float32)

        # Transpose if passed as (n_samples, channels)
        if chunk.ndim == 2 and chunk.shape[0] != self.in_channels and chunk.shape[1] == self.in_channels:
            chunk = chunk.T

        # Resample chunk if sampling rate differs from model operating rate (200 Hz)
        if self.fs != self.model_fs:
            up = int(self.model_fs)
            down = int(self.fs)
            div = gcd(up, down)
            chunk = signal.resample_poly(chunk, up // div, down // div, axis=-1).astype(np.float32)

        # Standardize if enabled
        if self.standardize_input:
            if self.mean is not None and self.std is not None:
                chunk = (chunk - self.mean) / self.std
            else:
                m = chunk.mean(axis=-1, keepdims=True)
                s = chunk.std(axis=-1, keepdims=True) + 1e-6
                chunk = (chunk - m) / s

        n_new = chunk.shape[1]
        if n_new >= self.sub_window_len:
            self.raw_buffer = chunk[:, -self.sub_window_len :]
        else:
            self.raw_buffer = np.roll(self.raw_buffer, -n_new, axis=-1)
            self.raw_buffer[:, -n_new:] = chunk

        # Feature Extraction: 300 ms sub-window -> CMTS vector
        sub_tensor = torch.from_numpy(self.raw_buffer).unsqueeze(0).to(self.device)
        cmts_feat = self._extract_single_cmts(sub_tensor)  # [1, total_feat_dim]
        self.feature_queue.append(cmts_feat)
        self.warmup_steps += 1

        # Assemble 10-step sequence (pad with oldest if queue not yet full)
        if len(self.feature_queue) < self.seq_len:
            pad_count = self.seq_len - len(self.feature_queue)
            seq_tensors = [self.feature_queue[0]] * pad_count + list(self.feature_queue)
        else:
            seq_tensors = list(self.feature_queue)

        seq_feats = torch.cat(seq_tensors, dim=0).unsqueeze(0)  # [1, 10, total_feat_dim]

        # Network Forward Pass
        h = self.model.input_dense(seq_feats)
        h, _ = self.model.gru1(h)
        h, _ = self.model.gru2(h)
        emb = self.model.hidden_dense(h[:, -1, :])
        logits = self.model.out(emb)

        if self.model.activation == "sigmoid":
            pred = torch.sigmoid(logits)
        elif self.model.activation == "tanh":
            pred = torch.tanh(logits)
        else:
            pred = logits

        pred_np = pred.squeeze(0).cpu().numpy()

        # Convert units if requested
        if self.units == "rad":
            pred_np = convert_angles(pred_np, from_unit="deg", to_unit="rad")

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return pred_np, latency_ms


# ==============================================================================
# Section 4: Offline Batch Recording Decoding
# ==============================================================================


def infer_recording(
    emg: np.ndarray,
    model: Model,
    cfg: Dict[str, Any],
    fs: float = DEFAULT_SAMPLING_RATE_HZ,
    units: str = "deg",
    device: Union[str, torch.device] = "auto",
    stride: Optional[int] = None,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Decode an entire continuous sEMG recording into joint angle trajectories.

    Args:
        emg: 2D array of sEMG signals of shape (channels, samples) or (samples, channels).
        model: Trained TRR Model instance.
        cfg: Pipeline configuration dictionary (from params.yaml).
        fs: Input sampling frequency in Hz (default: 200.0).
        units: Desired output units ('deg' or 'rad'). Defaults to 'deg'.
        device: 'auto', 'cuda', or 'cpu'.
        stride: Stride between successive predictions in samples. Defaults to dataset.stride or 20.
        batch_size: Batch size for GPU/CPU parallel forward passes. Defaults to 256.

    Returns:
        Tuple of (predicted_angles [N, 15], timestamps_sec [N], latency_stats).
    """
    t_start = time.perf_counter()
    emg_arr = np.asarray(emg, dtype=np.float32)
    in_channels = model.rfe.in_channels

    if emg_arr.ndim == 2 and emg_arr.shape[0] != in_channels and emg_arr.shape[1] == in_channels:
        emg_arr = emg_arr.T

    # 1. Conditioning & Resampling (if fs != 200)
    pre_cfg = cfg.get("preprocessing", {})
    downsample_factor = fs / MODEL_OPERATING_RATE_HZ if fs != MODEL_OPERATING_RATE_HZ else 1.0
    conditioned_emg = preprocess(
        emg_arr,
        fs=fs,
        bandpass=pre_cfg.get("bandpass", [15.0, 95.0]),
        notch=pre_cfg.get("notch", 50.0),
        downsample=downsample_factor,
        multiband=pre_cfg.get("multiband", False),
    )

    # 2. Per-subject standardization across time
    m = conditioned_emg.mean(axis=-1, keepdims=True)
    s = conditioned_emg.std(axis=-1, keepdims=True) + 1e-6
    normalized_emg = (conditioned_emg - m) / s

    # 3. Sliding Window Segmentation
    window_len = int(cfg.get("dataset", {}).get("window", 240))
    step = stride or int(cfg.get("dataset", {}).get("stride", 20))
    total_samples = normalized_emg.shape[1]

    if total_samples < window_len:
        raise ValueError(
            f"Recording has only {total_samples} samples at 200 Hz; " f"needs at least window length {window_len}."
        )

    starts = np.arange(0, total_samples - window_len + 1, step)
    windows = np.stack([normalized_emg[:, s_idx : s_idx + window_len] for s_idx in starts])

    # 4. Batched Model Forward Pass
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model.to(target_device)
    model.eval()

    preds_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_x = torch.from_numpy(windows[i : i + batch_size]).to(target_device)
            batch_out = model(batch_x).cpu().numpy()
            preds_list.append(batch_out)

    predictions = np.concatenate(preds_list, axis=0)

    # Convert units if requested
    if units.lower() == "rad":
        predictions = convert_angles(predictions, from_unit="deg", to_unit="rad")

    total_time = time.perf_counter() - t_start
    timestamps = (starts + window_len - 1) / MODEL_OPERATING_RATE_HZ

    stats = {
        "total_windows": len(windows),
        "total_time_sec": total_time,
        "throughput_windows_per_sec": len(windows) / max(total_time, 1e-6),
        "avg_latency_ms": (total_time / len(windows)) * 1000.0,
    }

    return predictions, timestamps, stats

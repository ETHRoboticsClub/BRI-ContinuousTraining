"""Temporal Riemannian Regressor (TRR) Model Architecture.

This module implements the TRR model described in Colot et al. (2026):
"Decoding High-Dimensional Finger Motion from EMG Using Riemannian Features and RNNs".

Sections:
1. Constants & Model Defaults: Architecture dimensions, shrinkage constants, and numerical epsilons.
2. Covariance Estimation & OAS Shrinkage: Vectorized Oracle Approximating Shrinkage.
3. Riemannian Tangent Space Mapping: Matrix logarithm projecting covariance to Euclidean space.
4. Riemannian Feature Extractor (RiemannianFeatureExtractor): Temporal sub-window CMTS extraction.
5. Recurrent Regressor (Model): Dense(256) -> Stacked GRU(256, 128) -> Dense(64) -> Output Head.
"""

from functools import partial
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.probes import dead_units, probe, saturation, temporal_dynamics

# ==============================================================================
# Section 1: Constants & Model Defaults
# ==============================================================================

# Architectural defaults
DEFAULT_HIDDEN_DENSE: int = 256
DEFAULT_GRU_UNITS: Sequence[int] = (256, 128)
DEFAULT_DENSE_UNITS: int = 64
DEFAULT_DROPOUT: float = 0.1
DEFAULT_N_SUB_WINDOWS: int = 10
DEFAULT_N_BANDS: int = 1
DEFAULT_ACTIVATION: str = "sigmoid"

# Riemannian geometry constants
RIEMANNIAN_ISOMETRY_OFF_DIAG_SCALE: float = float(np.sqrt(2.0))
RIEMANNIAN_ISOMETRY_DIAG_SCALE: float = 1.0

# Numerical stabilization and shrinkage parameters
OAS_DENOMINATOR_EPSILON: float = 1e-8
OAS_RHO_MIN: float = 0.0
OAS_RHO_MAX: float = 1.0
COVARIANCE_SPD_REGULARIZATION_EPSILON: float = 1e-5
EIGENVALUE_MIN_CLAMP: float = 1e-6

# Sub-window temporal slicing bounds
MIN_SUB_WINDOW_LENGTH: int = 2
MIN_STEP_LENGTH: int = 1
DEFAULT_STEP_MARGIN_FACTOR: int = 2


# ==============================================================================
# Section 2 & 3: Riemannian Feature Extractor
# ==============================================================================


class RiemannianFeatureExtractor(nn.Module):
    """Extracts temporal sequences of multi-band Riemannian Covariance in Tangent Space (CMTS) features.

    Accomplishes:
    1. Slices the input EMG window [B, C, W] into 10 overlapping temporal sub-windows
       (each 300 ms duration, with 100 ms step, strictly causal).
    2. Computes the spatial covariance matrix across channels for each sub-window and frequency band.
    3. Regularizes covariance matrices using Oracle Approximating Shrinkage (OAS).
    4. Projects regularized covariance matrices onto the Euclidean tangent space using the
       Riemannian matrix logarithm and an isometric basis (sqrt(2) on off-diagonal entries).
    5. Returns feature tensor [B, total_feat_dim, n_sub_windows] compatible with probe instrumentation.
    """

    def __init__(
        self,
        in_channels: int,
        n_bands: int = DEFAULT_N_BANDS,
        n_sub_windows: int = DEFAULT_N_SUB_WINDOWS,
        sub_window_len: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Initialize the RiemannianFeatureExtractor.

        Args:
            in_channels: Total number of input EMG channels (e.g. 8 for single band,
                or 24 for 3 frequency bands).
            n_bands: Number of frequency bands present in in_channels. Defaults to 1.
            n_sub_windows: Number of sliding temporal sub-windows to extract. Defaults to 10.
            sub_window_len: Duration of each sub-window in samples (e.g. 60 samples for
                300 ms at 200 Hz). If None, calculated automatically from window length.
            step: Stride between consecutive sub-windows in samples (e.g. 20 samples for
                100 ms at 200 Hz). If None, calculated automatically.
        """
        super().__init__()
        self.in_channels: int = in_channels
        if n_bands == 1 and in_channels > 8 and in_channels % 3 == 0:
            n_bands = 3

        self.n_bands: int = max(1, n_bands)
        if in_channels % self.n_bands != 0:
            # Fallback if in_channels is not evenly divisible: treat as single band
            self.ch_per_band: int = in_channels
            self.n_bands = 1
        else:
            self.ch_per_band = in_channels // self.n_bands

        self.n_sub_windows: int = n_sub_windows
        self.sub_window_len: Optional[int] = sub_window_len
        self.step: Optional[int] = step

        # Precompute upper triangular indices and Riemannian isometry weights
        num_ch: int = self.ch_per_band
        idx: Tuple[torch.Tensor, torch.Tensor] = torch.triu_indices(num_ch, num_ch)
        self.register_buffer("triu_row", idx[0])
        self.register_buffer("triu_col", idx[1])
        diag_mask: torch.Tensor = idx[0] == idx[1]
        weights: torch.Tensor = torch.where(
            diag_mask, RIEMANNIAN_ISOMETRY_DIAG_SCALE, RIEMANNIAN_ISOMETRY_OFF_DIAG_SCALE
        )
        self.register_buffer("triu_weights", weights)

        self.feat_dim_per_band: int = num_ch * (num_ch + 1) // 2
        self.total_feat_dim: int = self.feat_dim_per_band * self.n_bands

    def _oas_cov(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Oracle Approximating Shrinkage (OAS) regularized covariance matrices.

        Args:
            x: Input epoch tensor of shape [N, C, T] where N is batch size, C is channels,
                and T is sub-window sample count.

        Returns:
            Symmetric positive-definite regularized covariance tensor of shape [N, C, C].
        """
        _, num_ch, num_time = x.shape
        xc: torch.Tensor = x - x.mean(dim=-1, keepdim=True)
        s: torch.Tensor = (xc @ xc.transpose(-1, -2)) / max(1, num_time - 1)
        tr_s: torch.Tensor = torch.diagonal(s, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / num_ch
        tr_s2: torch.Tensor = (s * s).sum(dim=(-2, -1), keepdim=True)
        num: torch.Tensor = (1.0 - 2.0 / num_ch) * tr_s2 + (tr_s * num_ch) ** 2
        den: torch.Tensor = (num_time + 1.0 - 2.0 / num_ch) * (tr_s2 - (tr_s * num_ch) ** 2 / num_ch)
        rho: torch.Tensor = torch.clamp(num / (den + OAS_DENOMINATOR_EPSILON), OAS_RHO_MIN, OAS_RHO_MAX)
        eye: torch.Tensor = torch.eye(num_ch, device=x.device, dtype=x.dtype)
        cov: torch.Tensor = (1.0 - rho) * s + rho * tr_s * eye
        # Guarantee strict positive definiteness for matrix logarithm
        return cov + COVARIANCE_SPD_REGULARIZATION_EPSILON * eye

    def _tangent_space(self, cov: torch.Tensor) -> torch.Tensor:
        """Project SPD covariance matrices into the Euclidean tangent space at the Identity.

        Args:
            cov: Symmetric positive-definite covariance matrices of shape [N, C, C].

        Returns:
            Projected tangent feature vectors of shape [N, C*(C+1)//2] with isometry weighting.
        """
        evals: torch.Tensor
        evecs: torch.Tensor
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=EIGENVALUE_MIN_CLAMP)
        log_cov: torch.Tensor = evecs @ torch.diag_embed(torch.log(evals)) @ evecs.transpose(-1, -2)
        feats: torch.Tensor = log_cov[:, self.triu_row, self.triu_col] * self.triu_weights
        return feats

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-band CMTS sequence from raw EMG window.

        Args:
            x: Input EMG tensor of shape [B, C, W] where B is batch size, C is channels,
                and W is window sample length.

        Returns:
            Feature sequence tensor of shape [B, total_feat_dim, n_sub_windows].
        """
        batch_size, _, total_samples = x.shape
        step: int = self.step or max(1, total_samples // (self.n_sub_windows + DEFAULT_STEP_MARGIN_FACTOR))
        sw: int = self.sub_window_len or max(MIN_SUB_WINDOW_LENGTH, total_samples - (self.n_sub_windows - 1) * step)
        starts: List[int] = [
            total_samples - sw - (self.n_sub_windows - 1 - i) * step for i in range(self.n_sub_windows)
        ]
        subs: torch.Tensor = torch.stack([x[:, :, s : s + sw] for s in starts], dim=1)  # [B, S, C, sw]

        band_feats: List[torch.Tensor] = []
        for b in range(self.n_bands):
            xb: torch.Tensor = subs[:, :, b * self.ch_per_band : (b + 1) * self.ch_per_band, :].reshape(
                -1, self.ch_per_band, sw
            )
            cov: torch.Tensor = self._oas_cov(xb)
            tf: torch.Tensor = self._tangent_space(cov)
            band_feats.append(tf.view(batch_size, self.n_sub_windows, self.feat_dim_per_band))

        feats: torch.Tensor = torch.cat(band_feats, dim=-1)  # [B, S, total_feat_dim]
        # Return [B, total_feat_dim, S] for channel_activity & temporal_dynamics probes
        return feats.transpose(1, 2)


# ==============================================================================
# Section 4 & 5: Recurrent Regressor Architecture (TRR)
# ==============================================================================


class Model(nn.Module):
    """Temporal Riemannian Regressor (TRR) for high-dimensional finger motion regression.

    Accomplishes:
    1. Feature extraction: Multi-band Covariance in Tangent Space (CMTS) across 10 sub-windows.
    2. Input projection: Dense layer (256 units, tanh).
    3. Temporal context: Stacked GRUs (256 units -> 128 units, dropout=0.1).
    4. Representation embedding: Dense layer (64 units, tanh).
    5. Output decoding: Linear head mapping to target actuator activations / kinematics.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        hidden_dense: int = DEFAULT_HIDDEN_DENSE,
        gru_units: Sequence[int] = DEFAULT_GRU_UNITS,
        dense_units: int = DEFAULT_DENSE_UNITS,
        dropout: float = DEFAULT_DROPOUT,
        n_bands: int = DEFAULT_N_BANDS,
        n_sub_windows: int = DEFAULT_N_SUB_WINDOWS,
        sub_window_len: Optional[int] = None,
        step: Optional[int] = None,
        activation: str = DEFAULT_ACTIVATION,
        **kwargs: Any,
    ) -> None:
        """Initialize the TRR Model.

        Args:
            in_channels: Number of input channels (e.g. 8 for single band, 24 for 3-band).
            out_dim: Dimensionality of prediction target (e.g. 15 for finger joints, 3 for actuators).
            hidden_dense: Hidden units in the input projection dense layer. Defaults to 256.
            gru_units: Hidden units for the two stacked GRU layers. Defaults to (256, 128).
            dense_units: Dimensionality of the post-recurrent embedding dense layer. Defaults to 64.
            dropout: Dropout probability applied after each GRU layer. Defaults to 0.1.
            n_bands: Number of frequency bands present in in_channels. Defaults to 1.
            n_sub_windows: Number of sliding sub-windows. Defaults to 10.
            sub_window_len: Length of each sub-window in samples (or None for auto-fit).
            step: Stride between sub-windows in samples (or None for auto-fit).
            activation: Activation applied to predictions ('linear', 'sigmoid', or 'tanh').
                Defaults to 'sigmoid'.
            **kwargs: Absorbs legacy parameters (e.g. layers, kernel_size, hidden) for backward compatibility.
        """
        super().__init__()
        # If legacy 'hidden' argument was passed without hidden_dense
        if "hidden" in kwargs and hidden_dense == DEFAULT_HIDDEN_DENSE and kwargs["hidden"] != 32:
            hidden_dense = int(kwargs["hidden"])

        self.rfe = RiemannianFeatureExtractor(
            in_channels=in_channels,
            n_bands=n_bands,
            n_sub_windows=n_sub_windows,
            sub_window_len=sub_window_len,
            step=step,
        )
        feat_dim: int = self.rfe.total_feat_dim

        # TRR Architecture from Colot et al. (2026)
        self.input_dense = nn.Sequential(nn.Linear(feat_dim, hidden_dense), nn.Tanh())
        self.gru1 = nn.GRU(hidden_dense, gru_units[0], batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(gru_units[0], gru_units[1], batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.hidden_dense = nn.Sequential(nn.Linear(gru_units[1], dense_units), nn.Tanh())
        self.out = nn.Linear(dense_units, out_dim)
        self.activation: str = activation

    @probe(
        "features",
        "activation_stats",
        "channel_activity",
        partial(dead_units, unit_axis=1),
        partial(temporal_dynamics, time_axis=-1),
    )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-band CMTS feature sequence.

        Args:
            x: Input EMG tensor of shape [B, in_channels, window].

        Returns:
            Feature tensor of shape [B, total_feat_dim, n_sub_windows].
        """
        return self.rfe(x)

    @probe("embedding", "linear_probe", "target_cka", "effective_rank", "dead_units")
    @probe("embedding", "pca", every=5)
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Map feature sequence through input dense and stacked GRUs to latent embedding.

        Args:
            x: Input EMG tensor of shape [B, in_channels, window].

        Returns:
            Latent embedding tensor of shape [B, dense_units] (default 64-dimensional).
        """
        feats: torch.Tensor = self.encode(x)  # [B, feat_dim, n_steps]
        seq: torch.Tensor = feats.transpose(1, 2)  # [B, n_steps, feat_dim]
        h: torch.Tensor = self.input_dense(seq)
        h, _ = self.gru1(h)
        h = self.drop1(h)
        h, _ = self.gru2(h)
        h = self.drop2(h)
        last_step: torch.Tensor = h[:, -1, :]  # [B, gru_units[1]]
        return self.hidden_dense(last_step)

    @probe("output", saturation)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to decode kinematics or actuator activations.

        Args:
            x: Input EMG tensor of shape [B, in_channels, window].

        Returns:
            Model predictions of shape [B, out_dim].
        """
        emb: torch.Tensor = self.embed(x)
        logits: torch.Tensor = self.out(emb)
        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.activation == "tanh":
            return torch.tanh(logits)
        return logits

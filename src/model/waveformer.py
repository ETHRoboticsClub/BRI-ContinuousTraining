"""WaveFormer: A Lightweight Transformer Model for sEMG-based Gesture Recognition.

Implements the architecture from Chen et al. (arXiv:2506.11168v3) and
ForeverBlue816/WaveFormer (`PatchEmbed` + multi-level learnable `WTConv2d`
with high-frequency subband dropout + 2D channel/time positional embeddings +
6-layer `RoPEAttention` Transformer encoder + `[CLS]` token head).
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pywt as _pywt
except ImportError:  # pragma: no cover - optional dependency; pure-PyTorch db1 included
    _pywt = None


def trunc_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Truncated normal initializer matching timm.layers.trunc_normal_."""
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=mean + a * std, b=mean + b * std)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Compute 1D sine-cosine positional embeddings for positions ``pos``."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0**omega)
    pos_flat = pos.reshape(-1)
    out = np.einsum("m,d->md", pos_flat, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 1D sine-cosine positional embeddings of length ``grid_size`` (+1 if ``cls_token``)."""
    grid_w = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_w)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def _wavelet_coefficients(wave: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return ``(dec_lo, dec_hi, rec_lo, rec_hi)`` filter coefficients without requiring PyWavelets."""
    if _pywt is not None:
        w = _pywt.Wavelet(wave)
        return list(w.dec_lo), list(w.dec_hi), list(w.rec_lo), list(w.rec_hi)

    name = str(wave).lower()
    if name in ("db1", "haar"):
        s = 2.0**-0.5
        return [s, s], [-s, s], [s, s], [s, -s]
    if name == "db2":
        sq3 = math.sqrt(3.0)
        denom = 4.0 * math.sqrt(2.0)
        rec_lo = [(1.0 + sq3) / denom, (3.0 + sq3) / denom, (3.0 - sq3) / denom, (1.0 - sq3) / denom]
        rec_hi = [rec_lo[3], -rec_lo[2], rec_lo[1], -rec_lo[0]]
        dec_lo = rec_lo[::-1]
        dec_hi = rec_hi[::-1]
        return dec_lo, dec_hi, rec_lo, rec_hi
    raise ValueError(f"Unsupported wavelet {wave!r} without PyWavelets installed; use 'db1' (Haar) or 'db2'.")


class PatchEmbed(nn.Module):
    """Multi-variate sEMG signal to 2D patch embedding (``ForeverBlue816/WaveFormer/util/patch_embed.py``)."""

    def __init__(
        self,
        input_channels: int = 1,
        patch_size: tuple[int, int] = (1, 20),
        embed_dim: int = 256,
        norm_layer: type[nn.Module] | None = nn.LayerNorm,
        activation_fct: type[nn.Module] | None = nn.GELU,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.embed_dim = int(embed_dim)
        self.flatten = bool(flatten)

        self.proj = nn.Conv2d(input_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer is not None else nn.Identity()
        self.act_ft = activation_fct() if activation_fct is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, 1, C, T)`` -> ``(B, D, C', T')`` (or ``(B, N, D)`` if ``flatten=True``)."""
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act_ft(x)
        x = x.permute(0, 3, 1, 2)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


def create_learnable_wavelet_filter(
    wave: str,
    in_size: int,
    out_size: int,
    init_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Parameter, nn.Parameter]:
    """Initialize learnable 2D wavelet decomposition and reconstruction filters as ``nn.Parameter``s."""
    dec_lo_l, dec_hi_l, rec_lo_l, rec_hi_l = _wavelet_coefficients(wave)

    dec_hi = torch.tensor(dec_hi_l[::-1], dtype=dtype)
    dec_lo = torch.tensor(dec_lo_l[::-1], dtype=dtype)
    dec_filters_2d = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    dec_filters_2d = dec_filters_2d[:, None].repeat(in_size, 1, 1, 1) * float(init_scale)

    rec_hi = torch.tensor(rec_hi_l[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(rec_lo_l[::-1], dtype=dtype).flip(dims=[0])
    rec_filters_2d = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters_2d = rec_filters_2d[:, None].repeat(out_size, 1, 1, 1) * float(init_scale)

    return nn.Parameter(dec_filters_2d, requires_grad=True), nn.Parameter(rec_filters_2d, requires_grad=True)


def wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """2D depthwise wavelet decomposition splitting each channel into ``[LL, LH, HL, HH]``."""
    _b, c, _h, _w = x.shape
    pad = (filters.shape[-1] // 2, filters.shape[-2] // 2)
    out = F.conv2d(x, filters, bias=None, stride=2, groups=c, padding=pad)
    b_out, _c_out, new_h, new_w = out.shape
    return out.view(b_out, c, 4, new_h, new_w)


def inverse_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """2D depthwise inverse wavelet reconstruction from ``(B, C, 4, hh, ww)``."""
    b, c, _four, hh, ww = x.shape
    out = x.reshape(b, c * 4, hh, ww)
    pad = (filters.shape[3] // 2, filters.shape[2] // 2)
    return F.conv_transpose2d(out, filters, stride=2, groups=c, padding=pad)


class _ScaleModule(nn.Module):
    """Channel-wise learnable affine scaling parameter."""

    def __init__(self, dims: list[int], init_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * float(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


class WTConv2d(nn.Module):
    """Multi-level learnable 2D Wavelet Convolution with high-frequency subband dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 3,
        wt_type: str = "db1",
        init_scale: float = 1.0,
        highfreq_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        self.in_channels = int(in_channels)
        self.wt_levels = int(wt_levels)
        self.stride = int(stride)
        self.highfreq_dropout = float(highfreq_dropout)

        dec_filters, rec_filters = create_learnable_wavelet_filter(
            wave=wt_type,
            in_size=self.in_channels,
            out_size=self.in_channels,
            init_scale=init_scale,
        )
        self.wt_filter = dec_filters
        self.iwt_filter = rec_filters

        self.base_conv = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=self.in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, self.in_channels, 1, 1], init_scale=1.0)

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels * 4,
                    self.in_channels * 4,
                    kernel_size=kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=self.in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, self.in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels: list[torch.Tensor] = []
        x_h_in_levels: list[torch.Tensor] = []
        shapes_in_levels: list[torch.Size] = []

        curr_x_ll = x
        for lvl in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 != 0) or (curr_shape[3] % 2 != 0):
                pad_h = curr_shape[2] % 2
                pad_w = curr_shape[3] % 2
                curr_x_ll = F.pad(curr_x_ll, (0, pad_w, 0, pad_h))

            dec_out = wavelet_transform(curr_x_ll, self.wt_filter)
            out_reshape = dec_out.view(dec_out.shape[0], dec_out.shape[1] * 4, dec_out.shape[3], dec_out.shape[4])
            out_reshape = self.wavelet_convs[lvl](out_reshape)
            out_reshape = self.wavelet_scale[lvl](out_reshape)
            out_reshape = out_reshape.view(*dec_out.shape)

            lf = out_reshape[:, :, 0, :, :]
            hf = out_reshape[:, :, 1:4, :, :]
            if self.highfreq_dropout > 0.0 and self.training:
                hf = F.dropout(hf, p=self.highfreq_dropout, training=True)

            dec_mod = torch.cat([lf.unsqueeze(2), hf], dim=2)
            curr_x_ll = dec_mod[:, :, 0, :, :]
            x_ll_in_levels.append(curr_x_ll)
            x_h_in_levels.append(dec_mod[:, :, 1:4, :, :])

        next_x_ll: torch.Tensor | int = 0
        for lvl in reversed(range(self.wt_levels)):
            curr_x_ll = x_ll_in_levels[lvl] + next_x_ll
            curr_x_h = x_h_in_levels[lvl]
            curr_shape = shapes_in_levels[lvl]

            cat_4 = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            recon = inverse_wavelet_transform(cat_4, self.iwt_filter)
            next_x_ll = recon[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        x_base = self.base_scale(self.base_conv(x))
        return x_base + x_tag


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    """WaveletConv block: depthwise multi-level ``WTConv2d`` followed by ``1x1`` channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        wt_levels: int = 3,
        wt_type: str = "db1",
        highfreq_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.depthwise = WTConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            wt_levels=wt_levels,
            wt_type=wt_type,
            highfreq_dropout=highfreq_dropout,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DropPath(nn.Module):
    """Stochastic Depth drop path per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """Transformer feed-forward block matching timm.layers.Mlp."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) operating on ``(B, H, N, D)``."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = int(dim)
        self.base = float(base)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        b, h, n, d = x.shape
        assert d == self.dim and n == seq_len
        pos = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        freqs = pos[:, None] * inv_freq[None, :]
        cos_val = torch.cos(freqs).to(x.dtype).unsqueeze(1).unsqueeze(1)
        sin_val = torch.sin(freqs).to(x.dtype).unsqueeze(1).unsqueeze(1)

        x_reshape = x.permute(2, 0, 1, 3).reshape(n, b, h, d // 2, 2)
        x_even, x_odd = x_reshape[..., 0], x_reshape[..., 1]
        x_even_out = x_even * cos_val - x_odd * sin_val
        x_odd_out = x_even * sin_val + x_odd * cos_val
        x_out = torch.stack([x_even_out, x_odd_out], dim=-1).reshape(n, b, h, d)
        return x_out.permute(1, 2, 0, 3)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Positional Embeddings on Q and K."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        rope_dim: int | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = int(num_heads)
        self.dim = int(dim)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope_dim = self.head_dim if rope_dim is None else int(rope_dim)
        assert self.rope_dim % 2 == 0
        self.rope_q = RotaryEmbedding(self.rope_dim)
        self.rope_k = RotaryEmbedding(self.rope_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if 0 < self.rope_dim <= self.head_dim:
            q_rope, q_rem = q[..., : self.rope_dim], q[..., self.rope_dim :]
            k_rope, k_rem = k[..., : self.rope_dim], k[..., self.rope_dim :]
            q = torch.cat([self.rope_q(q_rope, seq_len=n), q_rem], dim=-1)
            k = torch.cat([self.rope_k(k_rope, seq_len=n), k_rem], dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        x_out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x_out))


class WaveFormerBlock(nn.Module):
    """Pre-Norm Transformer block using ``RoPEAttention`` and ``Mlp``."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * float(mlp_ratio)),
            act_layer=nn.GELU,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class WaveFormer(nn.Module):
    """WaveFormer sEMG gesture classifier (`PatchEmbed` + `WTConv2d` + `RoPE` Transformer)."""

    def __init__(
        self,
        channels: int = 8,
        num_classes: int = 6,
        window_size: int = 200,
        time_steps: int | None = None,
        patch_size: tuple[int, int] = (1, 20),
        input_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        wt_levels: int = 3,
        wt_type: str = "db1",
        wt_kernel_size: int = 3,
        highfreq_dropout: float = 0.1,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        norm_layer: type[nn.Module] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.window_size = int(time_steps if time_steps is not None else window_size)
        self.input_channels = int(input_channels)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.wt_levels = int(wt_levels)
        self.wt_type = str(wt_type)
        self.wt_kernel_size = int(wt_kernel_size)
        self.highfreq_dropout = float(highfreq_dropout)
        self.dropout = float(dropout)
        self.drop_path = float(drop_path)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        assert self.embed_dim % 2 == 0, "embed_dim must be even"

        self.patch_embed = PatchEmbed(
            input_channels=self.input_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten=False,
        )

        # Active WaveletConv module (Paper Section III-C & Eq. 5)
        self.wavelet_conv = DepthwiseSeparableConvWithWTConv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=self.wt_kernel_size,
            wt_levels=self.wt_levels,
            wt_type=self.wt_type,
            highfreq_dropout=self.highfreq_dropout,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # 2D spatial (Myo pod channel) + temporal patch positional embeddings
        self.grid_height = max(1, self.channels // self.patch_size[0])
        self.max_num_patches_x = max(1, self.window_size // self.patch_size[1])
        self.pos_embed_x = nn.Parameter(
            torch.zeros(1, self.max_num_patches_x + 1, self.embed_dim // 2), requires_grad=False
        )
        self.pos_embed_y = nn.Embedding(self.grid_height + 1, self.embed_dim // 2, padding_idx=0)

        dpr = np.linspace(0.0, self.drop_path, self.depth).tolist() if self.depth > 0 else []
        self.blocks = nn.ModuleList(
            [
                WaveFormerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    drop=self.dropout,
                    attn_drop=0.0,
                    drop_path=float(dpr[i]),
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)
        self.fc_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, self.num_classes)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize positional embeddings, patch projection, CLS token, Transformer layers, and head."""
        _pos_y = torch.randn(self.pos_embed_y.num_embeddings - 1, self.pos_embed_y.embedding_dim) * 0.02
        trunc_normal_(_pos_y, std=0.02)
        with torch.no_grad():
            self.pos_embed_y.weight[1:] = _pos_y

        _pos_x = get_1d_sincos_pos_embed(
            self.pos_embed_x.shape[-1],
            self.pos_embed_x.shape[-2] - 1,
            cls_token=True,
        )
        self.pos_embed_x.data.copy_(torch.from_numpy(_pos_x).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)
        trunc_normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input shapes ``(B, 1, C, T)``, ``(B, C, T)``, or ``(B, T, C)`` -> ``(B, 1, C, T)``."""
        if x.dim() == 4:
            return x
        if x.dim() == 3:
            if x.shape[1] == self.channels:
                return x.unsqueeze(1)
            if x.shape[2] == self.channels:
                return x.transpose(1, 2).unsqueeze(1)
        raise ValueError(
            f"Expected input shape (B, 1, {self.channels}, T), (B, {self.channels}, T), or (B, T, {self.channels}); "
            f"got {tuple(x.shape)}"
        )

    def forward_encoder_all_patches(
        self,
        x: torch.Tensor,
        pos_embed_y_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract sequence of ``[CLS]`` + ``C'*T'`` encoded tokens."""
        x_4d = self._prepare_input(x)
        x = self.patch_embed(x_4d)  # (B, D, Hp, Wp)
        x = self.wavelet_conv(x)  # (B, D, Hp, Wp)

        bsz, dim, hp, wp = x.shape

        # Build 2D channel + temporal positional embedding (Hp * Wp, D)
        if pos_embed_y_ids is None:
            y_ids = torch.arange(1, hp + 1, device=x.device, dtype=torch.long)
            pos_y = self.pos_embed_y(y_ids.clamp(max=self.grid_height))  # (Hp, D/2)
            pos_y = pos_y.unsqueeze(1).expand(hp, wp, -1).unsqueeze(0).expand(bsz, -1, -1, -1)
        else:
            pos_y = self.pos_embed_y(pos_embed_y_ids.clamp(max=self.grid_height))  # (B, Hp, Wp, D/2)

        if wp <= self.max_num_patches_x:
            pos_x_seq = self.pos_embed_x[:, 1 : 1 + wp, :]
        else:
            pos_x_seq = F.interpolate(
                self.pos_embed_x[:, 1:, :].transpose(1, 2),
                size=wp,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        pos_x = pos_x_seq.unsqueeze(1).expand(bsz, hp, wp, -1)  # (B, Hp, Wp, D/2)
        pos_2d = torch.cat([pos_y, pos_x], dim=-1).reshape(bsz, hp * wp, dim).to(dtype=x.dtype)

        x_tokens = x.flatten(2).transpose(1, 2) + pos_2d  # (B, Hp*Wp, D)

        cls_token = self.cls_token.expand(bsz, -1, -1)
        x_seq = torch.cat((cls_token, x_tokens), dim=1)  # (B, 1+N, D)

        for blk in self.blocks:
            x_seq = blk(x_seq)

        return self.norm(x_seq)

    def forward_features(self, x: torch.Tensor, pos_embed_y: torch.Tensor | None = None) -> torch.Tensor:
        """Return normalized ``[CLS]`` token embedding ``(B, D)``."""
        x_seq = self.forward_encoder_all_patches(x, pos_embed_y_ids=pos_embed_y)
        return x_seq[:, 0]

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, pos_embed_y: torch.Tensor | None = None) -> torch.Tensor:
        """Classify sEMG windows into ``(B, num_classes)`` logits."""
        emb = self.forward_features(x, pos_embed_y=pos_embed_y)
        return self.forward_head(emb)

    def predict_action_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Alias returning ``(B, num_classes)`` classification logits."""
        return self.forward(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities ``(B, num_classes)``."""
        return torch.softmax(self.forward(x), dim=-1)


def Waveformer_base(**kwargs: object) -> WaveFormer:
    """Construct WaveFormer-base configuration matching paper Section III (~3.1M params)."""
    defaults: dict[str, object] = {
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "mlp_ratio": 2.0,
        "wt_levels": 3,
        "wt_type": "db1",
        "wt_kernel_size": 3,
        "highfreq_dropout": 0.1,
    }
    defaults.update(kwargs)
    return WaveFormer(**defaults)  # type: ignore[arg-type]

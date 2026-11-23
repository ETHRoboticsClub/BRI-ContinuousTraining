"""Primary EMG classifier model exports (WaveFormer)."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.model.waveformer import WaveFormer, Waveformer_base

__all__ = [
    "Model",
    "WaveFormer",
    "Waveformer_base",
    "build_model_from_config",
]


def build_model_from_config(model_cfg: dict[str, Any], channels: int = 8) -> nn.Module:
    """Instantiate WaveFormer from a config dictionary."""
    arch = str(model_cfg.get("arch", "waveformer")).lower()
    if arch != "waveformer":
        raise ValueError(f"Unsupported model arch {arch!r}; expected 'waveformer'")

    patch_raw = model_cfg.get("patch_size", [1, 20])
    patch_size = (int(patch_raw[0]), int(patch_raw[1]))
    return WaveFormer(
        channels=int(model_cfg.get("channels", channels)),
        num_classes=int(model_cfg.get("num_classes", 6)),
        window_size=int(model_cfg.get("window_size", 200)),
        patch_size=patch_size,
        input_channels=int(model_cfg.get("input_channels", 1)),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 2.0)),
        wt_levels=int(model_cfg.get("wt_levels", 3)),
        wt_type=str(model_cfg.get("wt_type", "db1")),
        wt_kernel_size=int(model_cfg.get("wt_kernel_size", 3)),
        highfreq_dropout=float(model_cfg.get("highfreq_dropout", 0.1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        drop_path=float(model_cfg.get("drop_path", 0.1)),
    )


Model = WaveFormer

"""Unit tests for WaveFormer 6-class architecture."""

from __future__ import annotations

from pathlib import Path

import torch

from src.model.model import Model, WaveFormer, Waveformer_base, build_model_from_config
from src.trainer import load_waveformer_checkpoint, save_waveformer_checkpoint


def test_waveformer_alias_shapes_and_wavelet_gradients() -> None:
    assert Model is WaveFormer
    model = Waveformer_base(channels=8, num_classes=6, window_size=200, patch_size=(1, 20))
    assert sum(p.numel() for p in model.parameters()) > 1_000_000

    # Accept (B, 1, C, T), (B, C, T), and (B, T, C)
    x_4d = torch.randn(2, 1, 8, 200)
    x_3d_ct = x_4d.squeeze(1)
    x_3d_tc = x_3d_ct.transpose(1, 2)

    model.eval()
    with torch.no_grad():
        logits_4d = model(x_4d)
        logits_ct = model(x_3d_ct)
        logits_tc = model(x_3d_tc)
        probs = model.predict_proba(x_4d)

    assert logits_4d.shape == (2, 6)
    torch.testing.assert_close(logits_4d, logits_ct, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits_4d, logits_tc, atol=1e-5, rtol=1e-5)
    assert probs.shape == (2, 6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    # Confirm wavelet_conv is actively wired in the forward pass and receives gradients
    model.train()
    loss = model(x_4d).sum()
    loss.backward()
    assert model.wavelet_conv.depthwise.wt_filter.grad is not None
    assert model.wavelet_conv.pointwise.weight.grad is not None


def test_waveformer_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = {
        "arch": "waveformer",
        "channels": 8,
        "num_classes": 6,
        "window_size": 100,
        "patch_size": [1, 10],
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "wt_levels": 2,
    }
    model = build_model_from_config(cfg)
    model.eval()

    ckpt_path = save_waveformer_checkpoint(
        tmp_path / "waveformer.pt",
        model,
        args_dict=cfg,
        epoch=3,
        best_score=0.91,
        metrics={"val_acc": 0.91},
    )

    loaded, payload = load_waveformer_checkpoint(ckpt_path, device="cpu")
    loaded.eval()
    assert payload["arch"] == "waveformer"
    assert payload["args_dict"]["embed_dim"] == 64

    x = torch.randn(2, 1, 8, 100)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x), atol=1e-6, rtol=1e-6)

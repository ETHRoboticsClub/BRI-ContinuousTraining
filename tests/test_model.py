"""Unit tests for the 6-class ReactEMG Any2Any_Model architecture."""

from __future__ import annotations

from pathlib import Path

import torch

from src.model.model import Any2Any_Model, Model
from src.trainer import load_reactemg_checkpoint, save_reactemg_checkpoint


def test_model_alias_and_forward_shapes() -> None:
    assert Model is Any2Any_Model
    model = Any2Any_Model(num_classes=6, window_size=600, channels=8)
    model.eval()

    bsz = 3
    emg = torch.rand(bsz, 600, 8, dtype=torch.float32)
    actions = torch.randint(0, 7, (bsz, 600), dtype=torch.long)
    task_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    mask_pos_emg = torch.zeros(bsz, 600, 8, dtype=torch.bool)
    mask_pos_emg[1, 50:100, :] = True

    emg_out, act_out = model(emg, actions, task_idx, mask_pos_emg)
    assert emg_out.shape == (bsz, 600, 8)
    assert act_out.shape == (bsz, 600, 7)

    logits = model.predict_action_logits(emg)
    assert logits.shape == (bsz, 600, 6)


def test_shared_positional_encoding_and_task3_isolation() -> None:
    model = Any2Any_Model(num_classes=6, window_size=120, channels=8, share_pe=True)
    model.eval()
    assert model.emg_positional_encoding is model.action_positional_encoding

    emg = torch.rand(1, 120, 8, dtype=torch.float32)
    mask_pos_emg = torch.zeros(1, 120, 8, dtype=torch.bool)
    mask_pos_emg[:, 20:50, :] = True
    task_idx = torch.tensor([3], dtype=torch.long)

    actions_a = torch.full((1, 120), 6, dtype=torch.long)
    actions_b = torch.randint(0, 6, (1, 120), dtype=torch.long)

    with torch.no_grad():
        emg_out_a, _ = model(emg, actions_a, task_idx, mask_pos_emg)
        emg_out_b, _ = model(emg, actions_b, task_idx, mask_pos_emg)

    assert torch.isfinite(emg_out_a).all()
    assert torch.allclose(emg_out_a, emg_out_b, atol=1e-5)


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = Any2Any_Model(num_classes=6, window_size=120, embedding_dim=64, nhead=4, num_layers=1)
    model.eval()
    ckpt_path = save_reactemg_checkpoint(
        tmp_path / "reactemg.pt",
        model,
        args_dict={
            "num_classes": 6,
            "window_size": 120,
            "embedding_dim": 64,
            "nhead": 4,
            "num_layers": 1,
            "channels": 8,
        },
        epoch=1,
        best_val_loss=0.42,
    )

    loaded, payload = load_reactemg_checkpoint(ckpt_path, device="cpu")
    loaded.eval()
    assert payload["args_dict"]["num_classes"] == 6
    assert "model_info" in payload and "model_state_dict" in payload["model_info"]

    x = torch.rand(2, 120, 8)
    with torch.no_grad():
        assert torch.allclose(model.predict_action_logits(x), loaded.predict_action_logits(x), atol=1e-6)

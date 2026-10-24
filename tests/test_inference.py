"""Unit Tests for TRR Inference Engine (src/inference.py and scripts/infer_trr.py).

Verifies:
1. Model loading and configuration resolution.
2. Explicit sampling rate handling (default 200 Hz vs custom 500 Hz).
3. Streaming FIFO queue stateful execution (step latency, buffer rolling, warm-up).
4. Equivalence between streaming inference and full-window batch inference.
5. Unit conversions (degrees to radians and vice-versa).
6. Offline batch recording decoding and metric computations.
"""

import numpy as np
import pytest
import torch
import yaml

from src.inference import (
    DEFAULT_SAMPLING_RATE_HZ,
    MODEL_OPERATING_RATE_HZ,
    StreamingTRRInference,
    convert_angles,
    infer_recording,
    load_inference_artifacts,
)
from src.model import Model


@pytest.fixture
def dummy_model_and_config(tmp_path):
    """Create a temporary TRR model checkpoint and params.yaml for testing."""
    cfg = {
        "model": {
            "hidden_dense": 64,
            "gru_units": [64, 32],
            "dense_units": 32,
            "dropout": 0.1,
            "n_sub_windows": 10,
            "n_bands": 1,
            "activation": "linear",
        },
        "preprocessing": {
            "fs": 200,
            "bandpass": [15, 95],
            "notch": 50,
            "downsample": 1,
            "multiband": False,
        },
        "dataset": {
            "window": 240,
            "stride": 20,
        },
    }
    cfg_path = tmp_path / "params.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    model = Model(in_channels=8, out_dim=15, **cfg["model"])
    ckpt_path = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    return ckpt_path, cfg_path, cfg


def test_convert_angles():
    """Verify degree-to-radian and radian-to-degree mathematical accuracy."""
    deg_angles = np.array([0.0, 90.0, 180.0, 360.0])
    rad_expected = np.array([0.0, np.pi / 2, np.pi, 2 * np.pi])

    rad_actual = convert_angles(deg_angles, from_unit="deg", to_unit="rad")
    np.testing.assert_allclose(rad_actual, rad_expected, rtol=1e-5)

    deg_back = convert_angles(rad_actual, from_unit="rad", to_unit="deg")
    np.testing.assert_allclose(deg_back, deg_angles, rtol=1e-5)


def test_load_inference_artifacts(dummy_model_and_config):
    """Verify loading of model weights and architecture configuration."""
    ckpt_path, cfg_path, _ = dummy_model_and_config
    model, cfg, device = load_inference_artifacts(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        device="cpu",
    )
    assert isinstance(model, Model)
    assert not model.training
    assert device.type == "cpu"
    assert "model" in cfg


def test_streaming_inference_default_fs(dummy_model_and_config):
    """Verify StreamingTRRInference with default 200 Hz sampling rate."""
    ckpt_path, cfg_path, _ = dummy_model_and_config
    engine = StreamingTRRInference(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        fs=DEFAULT_SAMPLING_RATE_HZ,
        device="cpu",
        units="deg",
    )
    assert engine.fs == 200.0
    assert engine.sub_window_len == 60
    assert engine.step_len == 20

    # Stream 15 consecutive 100 ms chunks (20 samples @ 200 Hz)
    rng = np.random.default_rng(42)
    for _ in range(15):
        chunk = rng.standard_normal((8, 20)).astype(np.float32)
        angles, lat_ms = engine.step(chunk)
        assert angles.shape == (15,)
        assert np.all(np.isfinite(angles))
        assert lat_ms > 0.0

    assert len(engine.feature_queue) == 10


def test_streaming_inference_custom_fs(dummy_model_and_config):
    """Verify StreamingTRRInference with custom 500 Hz sampling rate and polyphase resampling."""
    ckpt_path, cfg_path, _ = dummy_model_and_config
    engine = StreamingTRRInference(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        fs=500.0,
        device="cpu",
        units="rad",
    )
    assert engine.fs == 500.0
    assert engine.model_fs == MODEL_OPERATING_RATE_HZ

    # 100 ms chunk @ 500 Hz = 50 samples
    rng = np.random.default_rng(42)
    for _ in range(12):
        chunk = rng.standard_normal((8, 50)).astype(np.float32)
        angles, lat_ms = engine.step(chunk)
        assert angles.shape == (15,)
        assert np.all(np.isfinite(angles))

    # Reset
    engine.reset()
    assert len(engine.feature_queue) == 0
    assert np.all(engine.raw_buffer == 0.0)


def test_streaming_vs_batch_equivalence(dummy_model_and_config):
    """Verify that stateful streaming FIFO feature forward pass produces identical results to full window pass."""
    ckpt_path, cfg_path, _ = dummy_model_and_config
    engine = StreamingTRRInference(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        fs=200.0,
        device="cpu",
        units="deg",
        standardize_input=False,
    )

    rng = np.random.default_rng(0)
    # 240 samples = full 1.2 s window
    window_data = rng.standard_normal((8, 240)).astype(np.float32)

    # 1. Full-window batch pass through model
    tensor_win = torch.from_numpy(window_data).unsqueeze(0)
    with torch.no_grad():
        expected_pred = engine.model(tensor_win).squeeze(0).numpy()

    # 2. Sequential feed into engine with subwindow slicing
    engine.reset()
    # 10 subwindows of 60 samples with step 20
    for i in range(10):
        s = i * 20
        sub = window_data[:, s : s + 60]
        # set buffer to subwindow and extract feature
        sub_t = torch.from_numpy(sub).unsqueeze(0)
        feat = engine._extract_single_cmts(sub_t)
        engine.feature_queue.append(feat)

    seq_feats = torch.cat(list(engine.feature_queue), dim=0).unsqueeze(0)
    with torch.no_grad():
        h = engine.model.input_dense(seq_feats)
        h, _ = engine.model.gru1(h)
        h, _ = engine.model.gru2(h)
        emb = engine.model.hidden_dense(h[:, -1, :])
        actual_pred = engine.model.out(emb).squeeze(0).numpy()

    np.testing.assert_allclose(actual_pred, expected_pred, atol=1e-5)


def test_infer_recording(dummy_model_and_config):
    """Verify offline batch decoding of full recordings."""
    ckpt_path, cfg_path, cfg = dummy_model_and_config
    model, _, _ = load_inference_artifacts(ckpt_path, cfg_path, device="cpu")

    rng = np.random.default_rng(1)
    # 600 samples at 200 Hz = 3 seconds
    emg = rng.standard_normal((8, 600)).astype(np.float32)

    preds, timestamps, stats = infer_recording(
        emg=emg,
        model=model,
        cfg=cfg,
        fs=200.0,
        units="deg",
        device="cpu",
        stride=20,
    )

    # window = 240, step = 20 -> (600 - 240) // 20 + 1 = 19 windows
    assert preds.shape == (19, 15)
    assert timestamps.shape == (19,)
    assert stats["total_windows"] == 19
    assert stats["throughput_windows_per_sec"] > 0

"""Testbench for Kernelized / KernelizedLinear, run against MetaTF and PyTorch.

Every expected value comes from the MetaTF fixtures in ``tests/fixtures`` (see make_kernelized_fixtures.py) or
from properties of MetaTF's layers. The float64 reference stored in the Kernelized fixtures sizes MetaTF's own
float32 rounding error (the floors in the forward/gradient tests) and is the expected value only in the
"no less accurate than MetaTF" checks; the fixture script gates it against MetaTF. No torch or TensorFlow import
happens here; the adapters in ``kernelized_backends`` do that lazily.
"""

from pathlib import Path

import numpy as np
import pytest
from kernelized_backends import BACKENDS, kernelized, projection, zoh
from numpy.testing import assert_allclose

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KERNELIZED_NAMES = ["small", "long", "model", "repeat3"]
PROJECTION_NAMES = ["small", "model", "head2d"]

backend_param = pytest.mark.parametrize("backend", BACKENDS)
kernelized_param = pytest.mark.parametrize("name", KERNELIZED_NAMES)
projection_param = pytest.mark.parametrize("name", PROJECTION_NAMES)


def load(prefix, name):
    with np.load(FIXTURES / f"{prefix}_{name}.npz") as f:
        return {k: f[k] for k in f.files}


def build_kernelized(backend, fx):
    return kernelized(
        backend,
        int(fx["in_channels"]),
        int(fx["out_channels"]),
        int(fx["length"]),
        int(fx["num_coeffs"]),
        int(fx["repeat"]),
        int(fx["speed"]),
        int(fx["speed_range"]),
    )


# ---------------------------------------------------------------- Kernelized


@backend_param
@kernelized_param
def test_init_matches_metatf(backend, name):
    fx = load("kernelized", name)
    w = build_kernelized(backend, fx).weights()
    for key in ("A", "B", "log_dt"):
        assert_allclose(w[key], fx[f"init_{key}"], rtol=1e-6, atol=1e-6, err_msg=key)
    assert w["C"].shape == fx["C"].shape


@backend_param
def test_C_init_scale(backend):
    C = kernelized(backend, 256, 128, 50, 32, 2, 4, 8).weights()["C"]
    assert abs(C.std() / np.sqrt(2 / 64) - 1) < 0.05


@backend_param
@kernelized_param
def test_zoh_matches_metatf(backend, name):
    fx = load("kernelized", name)
    a_hat, b_hat, dta = zoh(backend, fx["A"], fx["B"], fx["log_dt"])
    assert_allclose(a_hat, fx["zoh_A_hat"], rtol=1e-5, atol=1e-6)
    assert_allclose(b_hat, fx["zoh_B_hat"], rtol=1e-5, atol=1e-6)
    assert_allclose(dta, fx["zoh_dtA"], rtol=1e-5, atol=1e-6)


def assert_close(actual, expected, rtol, atol, floor, err_msg=""):
    """|actual - expected| <= max(atol + rtol*|expected|, floor) elementwise."""
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{err_msg}: shape {actual.shape} != {expected.shape}"
    excess = np.abs(actual - expected) - np.maximum(atol + rtol * np.abs(expected), floor)
    assert excess.max() <= 0, f"{err_msg}: max abs error {np.abs(actual - expected).max():.3e}, floor {floor:.3e}"


GRAD_KEYS = ("x", "A", "B", "C", "log_dt")


def configured_layer(backend, fx):
    layer = build_kernelized(backend, fx)
    layer.set_weights(fx["A"], fx["B"], fx["C"], fx["log_dt"])
    return layer


# Roundoff in float32/complex64 (phase arguments reach thousands of rad) makes MetaTF itself differ from the exact
# result by err_*. A backend within err_* of exact is within 2*err_* of MetaTF (triangle inequality): the 2x floors.


@backend_param
@kernelized_param
def test_forward_matches_metatf(backend, name):
    fx = load("kernelized", name)
    layer = configured_layer(backend, fx)
    assert_close(layer.forward(fx["x"]), fx["y_state"], 1e-4, 1e-5, 2 * fx["err_y_state"], "forward")
    assert_close(layer.forward_full(fx["x"]), fx["y_full"], 1e-4, 1e-5, 2 * fx["err_y_full"], "forward_full")


@backend_param
@kernelized_param
def test_forward_as_accurate_as_metatf(backend, name):
    fx = load("kernelized", name)
    layer = configured_layer(backend, fx)
    assert_close(layer.forward(fx["x"]), fx["y_exact"], 1e-4, 1e-5, fx["err_y_state"], "forward")
    assert_close(layer.forward_full(fx["x"]), fx["y_exact"], 1e-4, 1e-5, fx["err_y_full"], "forward_full")


@backend_param
@kernelized_param
def test_gradients_match_metatf(backend, name):
    fx = load("kernelized", name)
    grads = configured_layer(backend, fx).grads(fx["x"], fx["g"])
    for key in GRAD_KEYS:
        assert_close(grads[key], fx[f"grad_{key}"], 1e-3, 1e-4, 2 * fx[f"err_grad_{key}"], key)


@backend_param
@kernelized_param
def test_gradients_as_accurate_as_metatf(backend, name):
    fx = load("kernelized", name)
    grads = configured_layer(backend, fx).grads(fx["x"], fx["g"])
    for key in GRAD_KEYS:
        assert_close(grads[key], fx[f"grad_exact_{key}"], 1e-3, 1e-4, fx[f"err_grad_{key}"], key)


@backend_param
def test_causal(backend):
    layer = kernelized(backend, 6, 5, 10, 4, 2, 4, 8)
    rng = np.random.default_rng(1)
    # Inputs are scaled by 0.1: MetaTF's float32 FFT path leaks ~3e-6 roundoff on unit-scale inputs (outputs ~15),
    # which would trip the 1e-6 tolerance without any real causality violation. A real leak would be O(0.1).
    x = 0.1 * rng.standard_normal((2, 10, 6)).astype(np.float32)
    x2 = x.copy()
    x2[:, 7:] += 0.1 * rng.standard_normal(x2[:, 7:].shape).astype(np.float32)
    y, y2 = layer.forward(x), layer.forward(x2)
    assert_allclose(y[:, :7], y2[:, :7], atol=1e-6)
    # the perturbation must be visible at and after step 7, so a constant (e.g. zero) output cannot pass
    assert np.abs(y2[:, 7:] - y[:, 7:]).max() > 1e-3


@backend_param
def test_parameters_are_A_B_C_log_dt(backend):
    assert kernelized(backend, 6, 5, 10, 4, 2, 4, 8).param_names() == {"A", "B", "C", "log_dt"}


# ---------------------------------------------------------------- KernelizedLinear vs StatefulProjection


@backend_param
@projection_param
def test_linear_forward_matches_metatf(backend, name):
    fx = load("projection", name)
    layer = projection(backend, int(fx["in_channels"]), int(fx["out_channels"]))
    layer.set_weights(fx["C"])
    assert_allclose(layer.forward(fx["x"]), fx["y"], rtol=1e-5, atol=1e-6)


@backend_param
@projection_param
def test_linear_gradients_match_metatf(backend, name):
    fx = load("projection", name)
    layer = projection(backend, int(fx["in_channels"]), int(fx["out_channels"]))
    layer.set_weights(fx["C"])
    grads = layer.grads(fx["x"], fx["g"])
    assert_allclose(grads["x"], fx["grad_x"], rtol=1e-4, atol=1e-5)
    assert_allclose(grads["C"], fx["grad_C"], rtol=1e-4, atol=1e-5)


@backend_param
def test_linear_memoryless(backend):
    layer = projection(backend, 6, 5)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 8, 6)).astype(np.float32)
    x2 = x.copy()
    x2[:, 3] += 1.0
    diff = np.abs(layer.forward(x2) - layer.forward(x)).max(axis=(0, 2))
    assert diff[3] > 1e-3
    assert_allclose(np.delete(diff, 3), 0.0, atol=1e-7)


@backend_param
def test_linear_only_C(backend):
    assert projection(backend, 6, 5).param_names() == {"C"}


@backend_param
def test_linear_init_glorot(backend):
    C = projection(backend, 256, 512).weights()["C"]
    a = np.sqrt(6 / (256 + 512))
    assert np.abs(C).max() <= a
    assert abs(C.std() / (a / np.sqrt(3)) - 1) < 0.05

"""Generate the MetaTF reference fixtures for the Kernelized / KernelizedLinear testbench.

Run ONLY in a throwaway MetaTF environment, never in the project env. This script is never imported by tests
or CI; its output (``tests/fixtures/*.npz``) is committed.

Environment setup (outside the repo)::

    python -m venv ~/.venvs/metatf
    ~/.venvs/metatf/Scripts/pip install "akida_models==1.14.2" "tensorflow~=2.19.0" pytest

If ``akida_models`` fails to install on Windows, install ``"tensorflow~=2.19.0" "tf_keras~=2.19.0" scipy pytest``,
run ``pip download akida_models==1.14.2 quantizeml==1.2.5 --no-deps`` into a scratch dir, extract both, set
``METATF_CUSTOM_LAYERS`` to ``akida_models/custom_layers.py`` and either set ``METATF_STATEFUL_RECURRENT`` to
``quantizeml/layers/stateful_recurrent.py`` (if it imports standalone) or ``pip install quantizeml==1.2.5``.

Then::

    ~/.venvs/metatf/Scripts/python tests/fixtures/make_kernelized_fixtures.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from kernelized_backends import kernelized, projection, zoh  # noqa: E402

REL_ERR_GATE = 1e-4

SOURCE = "akida_models 1.14.2 custom_layers.Kernelized; quantizeml 1.2.5 StatefulProjection(use_bias=False)"

# name: (in, out, num_coeffs, repeat, speed, speed_range, length, batch)
KERNELIZED = {
    "small": (6, 5, 4, 2, 4, 8, 10, 3),
    "long": (16, 8, 32, 2, 4, 8, 50, 2),
    "model": (256, 128, 32, 2, 4, 8, 50, 4),
    "repeat3": (5, 7, 8, 3, 2, 4, 20, 2),
}
# name: (in, out, x shape)
PROJECTION = {
    "small": (6, 5, (3, 10, 6)),
    "model": (36, 256, (4, 50, 36)),
    "head2d": (64, 15, (4, 64)),
}


def seed():
    import tensorflow as tf

    np.random.seed(0)
    tf.random.set_seed(0)


def exact_forward_and_grads(A, B, C, log_dt, x, g):
    """Kernelized forward pass and gradients of sum(y * g) in float64/complex128, from MetaTF's own math.

    y[b,t] = sum_{s<=t} x[b,s] @ (B_hat * Re(exp((t-s) * dtA))) @ C, with dt = exp(log_dt), dtA = dt * (-exp(A[:,0])
    + i A[:,1]) and B_hat = dt * B (zoh_discretize). The convolution is a direct sum over lags, not an FFT, so
    this shares no code path with either backend.
    """
    import tensorflow as tf

    v = {k: tf.Variable(a.astype(np.float64)) for k, a in dict(A=A, B=B, C=C, log_dt=log_dt).items()}
    xt = tf.constant(x.astype(np.float64))
    gt = tf.constant(g.astype(np.float64))
    length = x.shape[1]
    with tf.GradientTape() as tape:
        tape.watch(xt)
        dt = tf.exp(v["log_dt"])
        dtA = tf.cast(dt, tf.complex128) * tf.complex(-tf.exp(v["A"][:, 0]), v["A"][:, 1])
        b_hat = dt * v["B"]
        y = tf.zeros((x.shape[0], length, C.shape[1]), tf.float64)
        for lag in range(length):
            k = tf.math.real(tf.exp(tf.cast(lag, tf.complex128) * dtA))
            kernel = tf.einsum("cn,nd,n->cd", b_hat, v["C"], k)
            shifted = tf.einsum("blc,cd->bld", xt[:, : length - lag], kernel)
            y += tf.pad(shifted, [[0, 0], [lag, 0], [0, 0]])
        loss = tf.reduce_sum(y * gt)
    names = ["A", "B", "C", "log_dt"]
    grads = tape.gradient(loss, [xt] + [v[n] for n in names])
    return y.numpy(), {k: gr.numpy() for k, gr in zip(["x"] + names, grads, strict=True)}


def make_kernelized(name, cfg):
    cin, cout, num_coeffs, repeat, speed, speed_range, length, batch = cfg
    seed()
    layer = kernelized("metatf", cin, cout, length, num_coeffs, repeat, speed, speed_range)
    init = layer.weights()
    rng = np.random.default_rng(0)
    w = {k: (v + 0.1 * rng.standard_normal(v.shape)).astype(np.float32) for k, v in init.items()}
    layer.set_weights(**w)
    x = rng.standard_normal((batch, length, cin)).astype(np.float32)
    g = rng.standard_normal((batch, length, cout)).astype(np.float32)
    grads = layer.grads(x, g)
    a_hat, b_hat, dta = zoh("metatf", w["A"], w["B"], w["log_dt"])
    y_state, y_full = layer.forward(x), layer.forward_full(x)
    y_exact, grad_exact = exact_forward_and_grads(w["A"], w["B"], w["C"], w["log_dt"], x, g)
    # MetaTF's own float32 rounding error, and the gate tying the float64 reference to MetaTF: a wrong formula
    # would give an O(1) relative error here
    errs = {
        "err_y_state": (np.abs(y_state - y_exact).max(), np.abs(y_exact).max()),
        "err_y_full": (np.abs(y_full - y_exact).max(), np.abs(y_exact).max()),
        **{f"err_grad_{k}": (np.abs(grads[k] - grad_exact[k]).max(), np.abs(grad_exact[k]).max()) for k in grads},
    }
    for key, (err, scale) in errs.items():
        print(f"{name}: {key} = {err:.3e} (relative {err / scale:.3e})")
    bad = [k for k, (err, scale) in errs.items() if not err / scale < REL_ERR_GATE]
    if bad:
        raise RuntimeError(f"{name}: float64 reference disagrees with MetaTF beyond {REL_ERR_GATE}: {bad}")
    out = {
        "in_channels": cin,
        "out_channels": cout,
        "num_coeffs": num_coeffs,
        "repeat": repeat,
        "speed": speed,
        "speed_range": speed_range,
        "length": length,
        "batch": batch,
        "init_A": init["A"],
        "init_B": init["B"],
        "init_log_dt": init["log_dt"],
        **w,
        "x": x,
        "g": g,
        "y_state": y_state,
        "y_full": y_full,
        **{f"grad_{k}": v for k, v in grads.items()},
        # stored as float32 to keep the file under 1 MB; the rounding (~6e-8 relative) is far below MetaTF's error
        "y_exact": y_exact.astype(np.float32),
        **{f"grad_exact_{k}": v.astype(np.float32) for k, v in grad_exact.items()},
        **{k: np.float64(err) for k, (err, _) in errs.items()},
        "zoh_A_hat": a_hat,
        "zoh_B_hat": b_hat,
        "zoh_dtA": dta,
        "source": SOURCE,
    }
    np.savez_compressed(HERE / f"kernelized_{name}.npz", **out)


def make_projection(name, cfg):
    cin, cout, xshape = cfg
    seed()
    layer = projection("metatf", cin, cout)
    c = layer.weights()["C"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(xshape).astype(np.float32)
    g = rng.standard_normal((*xshape[:-1], cout)).astype(np.float32)
    grads = layer.grads(x, g)
    np.savez_compressed(
        HERE / f"projection_{name}.npz",
        in_channels=cin,
        out_channels=cout,
        C=c,
        x=x,
        g=g,
        y=layer.forward(x),
        grad_x=grads["x"],
        grad_C=grads["C"],
        source=SOURCE,
    )


if __name__ == "__main__":
    for n, c in KERNELIZED.items():
        make_kernelized(n, c)
    for n, c in PROJECTION.items():
        make_projection(n, c)
    for p in sorted(HERE.glob("*.npz")):
        print(p.name, p.stat().st_size)

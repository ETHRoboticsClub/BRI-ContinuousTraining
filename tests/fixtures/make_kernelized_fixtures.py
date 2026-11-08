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
        "y_state": layer.forward(x),
        "y_full": layer.forward_full(x),
        **{f"grad_{k}": v for k, v in grads.items()},
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

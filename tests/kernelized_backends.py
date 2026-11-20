"""Backend adapters for the Kernelized / KernelizedLinear testbench.

Two backends expose the same NumPy-in, NumPy-out interface:

- ``metatf``: the real ``akida_models.custom_layers.Kernelized`` and
  ``quantizeml.layers.StatefulProjection(use_bias=False)``.
- ``torch``: ``src.model.kernelized.Kernelized`` / ``KernelizedLinear`` / ``zoh_discretize``.

TensorFlow and torch are imported lazily inside the adapters, so this module imports cleanly in an environment
that has only one of them. All inputs and outputs are float32 (complex64 for complex values).
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_CUSTOM_LAYERS_ENV = "METATF_CUSTOM_LAYERS"
_STATEFUL_RECURRENT_ENV = "METATF_STATEFUL_RECURRENT"


def _metatf_available():
    if os.environ.get(_CUSTOM_LAYERS_ENV) and os.environ.get(_STATEFUL_RECURRENT_ENV):
        return True
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("akida_models", "quantizeml"))
    except (ImportError, ValueError):
        return False


BACKENDS = ["torch"] + (["metatf"] if _metatf_available() else [])


def _load_from_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _metatf_custom_layers():
    path = os.environ.get(_CUSTOM_LAYERS_ENV)
    if path:
        return _load_from_file("metatf_custom_layers", path)
    import akida_models.custom_layers as module

    return module


def _metatf_projection_class():
    path = os.environ.get(_STATEFUL_RECURRENT_ENV)
    if path:
        try:
            return _load_from_file("metatf_stateful_recurrent", path).StatefulProjection
        except ImportError:
            pass  # relative imports do not resolve standalone; fall back to the installed package
    from quantizeml.layers import StatefulProjection

    return StatefulProjection


def _short(name):
    """Keras weight name ('kernelized/A:0') to its last path part ('A')."""
    return name.split("/")[-1].split(":")[0]


# --------------------------------------------------------------------------------------------------------------
# MetaTF (TensorFlow / Keras)
# --------------------------------------------------------------------------------------------------------------


class _MetaTFKernelized:
    def __init__(self, in_channels, out_channels, length, num_coeffs, repeat, speed, speed_range):
        self.layer = _metatf_custom_layers().Kernelized(
            num_coeffs=num_coeffs, out_channels=out_channels, repeat=repeat, speed=speed, speed_range=speed_range
        )
        self.layer.build((None, length, in_channels))

    def _by_name(self):
        return {_short(w.name): w for w in self.layer.weights}

    def param_names(self):
        return set(self._by_name())

    def weights(self):
        return {k: v.numpy() for k, v in self._by_name().items()}

    def set_weights(self, A, B, C, log_dt):
        by_name = self._by_name()
        for name, value in dict(A=A, B=B, C=C, log_dt=log_dt).items():
            by_name[name].assign(np.asarray(value, dtype=np.float32))

    def _call(self, xt):
        """Call the layer as MetaTF does inside a model: traced with a dynamic batch dimension.

        Eagerly, `_padded_fft_corr` fails its static shape check when batch happens to differ from in_channels.
        """
        import tensorflow as tf

        spec = tf.TensorSpec((None, *xt.shape[1:]), tf.float32)
        return tf.function(self.layer, input_signature=[spec])(xt)

    def forward(self, x):
        import tensorflow as tf

        return self._call(tf.constant(x, dtype=tf.float32)).numpy()

    def forward_full(self, x):
        self.layer.force_full_conv = True
        try:
            return self.forward(x)
        finally:
            self.layer.force_full_conv = False

    def grads(self, x, g):
        import tensorflow as tf

        xt = tf.constant(x, dtype=tf.float32)
        gt = tf.constant(g, dtype=tf.float32)
        by_name = self._by_name()
        names = ["A", "B", "C", "log_dt"]
        with tf.GradientTape() as tape:
            tape.watch(xt)
            loss = tf.reduce_sum(self._call(xt) * gt)
        grads = tape.gradient(loss, [xt] + [by_name[n] for n in names])
        return {k: v.numpy() for k, v in zip(["x"] + names, grads, strict=True)}


class _MetaTFProjection:
    def __init__(self, in_channels, out_channels):
        self.layer = _metatf_projection_class()(out_channels, use_bias=False)
        self.layer.build((None, None, in_channels))

    def param_names(self):
        return {"C" if _short(w.name) == "kernel" else _short(w.name) for w in self.layer.weights}

    def weights(self):
        return {"C": self.layer.kernel.numpy()}

    def set_weights(self, C):
        self.layer.kernel.assign(np.asarray(C, dtype=np.float32))

    def forward(self, x):
        import tensorflow as tf

        return self.layer(tf.constant(x, dtype=tf.float32)).numpy()

    def grads(self, x, g):
        import tensorflow as tf

        xt = tf.constant(x, dtype=tf.float32)
        gt = tf.constant(g, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xt)
            loss = tf.reduce_sum(self.layer(xt) * gt)
        gx, gc = tape.gradient(loss, [xt, self.layer.kernel])
        return {"x": gx.numpy(), "C": gc.numpy()}


# --------------------------------------------------------------------------------------------------------------
# PyTorch (src.model.kernelized)
# --------------------------------------------------------------------------------------------------------------


class _TorchAdapter:
    """Shared logic: a torch module whose parameters are named like the Keras weights."""

    def __init__(self, module):
        self.module = module

    def param_names(self):
        return {name for name, _ in self.module.named_parameters()}

    def weights(self):
        return {name: p.detach().cpu().numpy() for name, p in self.module.named_parameters()}

    def _assign(self, **values):
        import torch

        params = dict(self.module.named_parameters())
        with torch.no_grad():
            for name, value in values.items():
                params[name].copy_(torch.from_numpy(np.asarray(value, dtype=np.float32)))

    def forward(self, x):
        import torch

        with torch.no_grad():
            return self.module(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()

    def grads(self, x, g):
        import torch

        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).requires_grad_(True)
        gt = torch.from_numpy(np.asarray(g, dtype=np.float32))
        self.module.zero_grad(set_to_none=True)
        (self.module(xt) * gt).sum().backward()
        out = {"x": xt.grad.numpy()}
        for name, p in self.module.named_parameters():
            out[name] = p.grad.numpy()
        return out


class _TorchKernelized(_TorchAdapter):
    def __init__(self, in_channels, out_channels, length, num_coeffs, repeat, speed, speed_range):
        from src.model.kernelized import Kernelized

        super().__init__(
            Kernelized(
                in_channels=in_channels,
                out_channels=out_channels,
                length=length,
                num_coeffs=num_coeffs,
                repeat=repeat,
                speed=speed,
                speed_range=speed_range,
            )
        )

    def set_weights(self, A, B, C, log_dt):
        self._assign(A=A, B=B, C=C, log_dt=log_dt)

    def forward_full(self, x):
        return self.forward(x)


class _TorchProjection(_TorchAdapter):
    def __init__(self, in_channels, out_channels):
        from src.model.kernelized import KernelizedLinear

        super().__init__(KernelizedLinear(in_channels, out_channels))

    def set_weights(self, C):
        self._assign(C=C)


# --------------------------------------------------------------------------------------------------------------
# Public factories
# --------------------------------------------------------------------------------------------------------------


def _check(backend):
    if backend not in ("metatf", "torch"):
        raise ValueError(f"unknown backend {backend!r}")


def kernelized(backend, in_channels, out_channels, length, num_coeffs, repeat, speed, speed_range):
    _check(backend)
    cls = _MetaTFKernelized if backend == "metatf" else _TorchKernelized
    return cls(in_channels, out_channels, length, num_coeffs, repeat, speed, speed_range)


def projection(backend, in_channels, out_channels):
    _check(backend)
    cls = _MetaTFProjection if backend == "metatf" else _TorchProjection
    return cls(in_channels, out_channels)


def zoh(backend, A, B, log_dt):
    """Return (A_hat complex64, B_hat float32, dtA complex64) as NumPy arrays."""
    _check(backend)
    if backend == "metatf":
        import tensorflow as tf

        outs = _metatf_custom_layers().zoh_discretize(
            tf.constant(A, tf.float32), tf.constant(B, tf.float32), tf.constant(log_dt, tf.float32)
        )
        return tuple(o.numpy() for o in outs)
    import torch
    from src.model.kernelized import zoh_discretize

    with torch.no_grad():
        outs = zoh_discretize(
            torch.from_numpy(np.asarray(A, np.float32)),
            torch.from_numpy(np.asarray(B, np.float32)),
            torch.from_numpy(np.asarray(log_dt, np.float32)),
        )
    return tuple(o.numpy() for o in outs)

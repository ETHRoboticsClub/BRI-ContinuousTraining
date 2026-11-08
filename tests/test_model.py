import pytest
import torch
from torch import nn

from src.model.model import Model
from src.model.tangent_space import n_steps, raw_length


def test_step_helpers():
    assert (n_steps(240), n_steps(1040), n_steps(64), n_steps(59)) == (10, 50, 1, 0)
    assert (raw_length(10), raw_length(50)) == (240, 1040)


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_model_output_shape(k, act):
    model = Model(8, 15, 10 * k, act)
    assert model(torch.randn(4, 8, raw_length(10 * k))).shape == (4, 15)


def test_model_has_no_bias():
    names = [n for n, _ in Model(8, 15, 10).named_parameters()]
    assert not any("bias" in n for n in names)


def test_activation_variant_is_used_everywhere():
    for act, cls in [("tanh", nn.Tanh), ("relu", nn.ReLU)]:
        mods = list(Model(8, 15, 10, act).modules())
        assert sum(isinstance(m, cls) for m in mods) == 4
        assert not any(isinstance(m, nn.Tanh if cls is nn.ReLU else nn.ReLU) for m in mods)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        Model(8, 15, 10, "gelu")


def test_model_rejects_wrong_length():
    with pytest.raises(ValueError):
        Model(8, 15, 50)(torch.randn(2, 8, 240))


def test_layer_widths():
    m = Model(8, 15, 10)
    shapes = (m.encoder.proj.C.shape, m.encoder.ssm1.C.shape, m.encoder.ssm2.C.shape)
    assert shapes == ((36, 256), (64, 256), (64, 128))
    assert (m.head[0].C.shape, m.head[2].C.shape) == ((128, 64), (64, 15))

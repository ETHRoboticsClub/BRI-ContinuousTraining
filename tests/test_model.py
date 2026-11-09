import pytest
import torch
from torch import nn

from src.model.kernelized import Kernelized
from src.model.model import Model
from src.model.ssm import SSMEncoder
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


@pytest.mark.parametrize("L,s", [(10, 2), (10, 5), (50, 2), (50, 5)])
def test_downsample_keeps_output_shape(L, s):
    assert Model(8, 15, L, downsample=s)(torch.randn(4, 8, raw_length(L))).shape == (4, 15)


def test_downsample_feeds_every_sth_step_to_ssm2():
    enc = SSMEncoder(36, 10, "tanh", downsample=2).eval()
    seen = {}
    enc.ssm1.register_forward_hook(lambda m, i, o: seen.update(o1=o))
    enc.ssm2.register_forward_hook(lambda m, i, o: seen.update(i2=i[0]))
    enc(torch.randn(3, 10, 36))
    torch.testing.assert_close(seen["i2"], torch.tanh(seen["o1"][:, 1::2]))


def test_downsample_inits_ssm2_from_reduced_length():
    enc = SSMEncoder(36, 50, "tanh", downsample=5)
    torch.testing.assert_close(enc.ssm2.log_dt, Kernelized(256, 128, 10).log_dt)
    torch.testing.assert_close(enc.ssm1.log_dt, Kernelized(256, 256, 50).log_dt)


@pytest.mark.parametrize("L,s", [(10, 3), (10, 10), (10, 0)])
def test_bad_downsample_raises(L, s):
    with pytest.raises(ValueError):
        SSMEncoder(36, L, "tanh", downsample=s)


def test_downsample_1_is_baseline():
    x = torch.randn(2, 8, 240)
    torch.manual_seed(0)
    a = Model(8, 15, 10).eval()
    torch.manual_seed(0)
    b = Model(8, 15, 10, downsample=1).eval()
    for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert na == nb and torch.equal(pa, pb)
    torch.testing.assert_close(a(x), b(x))

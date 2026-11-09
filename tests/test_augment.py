import pytest
import torch

from src.augment import Augment, channel_gain

ZEROS = {"aug_gain": 0.0, "aug_rotate": 0, "aug_shift": 0.0}


def test_gain_zero_is_identity():
    x = torch.randn(8, 240)
    assert torch.equal(channel_gain(x, torch.zeros(8)), x)
    assert Augment.from_params(ZEROS) is None
    assert isinstance(Augment.from_params({**ZEROS, "aug_gain": 0.2}), Augment)


def test_gain_is_per_channel_and_constant_over_time():
    aug = Augment(gain=0.2)
    aug.manual_seed(0)
    x = torch.rand(8, 240) + 0.5
    ratio = aug(x) / x
    torch.testing.assert_close(ratio, ratio[:, :1].expand_as(ratio))  # one gain per channel
    assert ratio[:, 0].unique().numel() == 8  # different across channels


def test_gain_strength():
    aug = Augment(gain=0.2)
    aug.manual_seed(0)
    log_gains = torch.cat([aug(torch.ones(8, 1)).log().flatten() for _ in range(2000)])
    assert abs(log_gains.std().item() - 0.2) < 0.01


def test_augment_is_reproducible():
    a, b = Augment(gain=0.2), Augment(gain=0.2)
    a.manual_seed(3)
    b.manual_seed(3)
    x = torch.randn(8, 240)
    for _ in range(3):
        assert torch.equal(a(x), b(x))


def test_augment_leaves_global_rng_alone():
    aug = Augment(gain=0.2)
    aug.manual_seed(0)
    x = torch.ones(8, 240)
    torch.manual_seed(1)
    a = torch.rand(3)
    torch.manual_seed(1)
    aug(x)
    b = torch.rand(3)
    assert torch.equal(a, b)


@pytest.mark.parametrize("kwargs", [{"gain": -0.1}, {"rotate": -1}, {"shift": -0.5}])
def test_negative_strengths_raise(kwargs):
    with pytest.raises(ValueError):
        Augment(**kwargs)

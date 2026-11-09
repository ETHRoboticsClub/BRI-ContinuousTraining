import pytest
import torch

from src.augment import Augment, channel_gain, rotate, shift

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


def test_rotate_is_roll_in_ring_order():
    x = torch.arange(8.0)[:, None].expand(8, 5)
    torch.testing.assert_close(rotate(x, 1, order=tuple(range(8))), x.roll(1, 0))
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    y = rotate(x, 1, order=order)
    assert [int(y[c, 0]) for c in order] == [order[(i - 1) % 8] for i in range(8)]


def test_rotate_full_turn_is_identity():
    x = torch.randn(8, 20)
    assert torch.equal(rotate(x, 8), x)
    assert torch.equal(rotate(x, -1), rotate(x, 7))


def test_rotation_draws_cover_minus_R_to_R():
    aug = Augment(rotate=1)
    aug.manual_seed(0)
    x = torch.arange(8.0)[:, None]
    seen = {int(aug(x)[0, 0].item()) for _ in range(300)}  # channel 0 receives channel -r
    assert {(-v) % 8 for v in seen} == {0, 1, 7}


def test_shift_integer_values_are_rotations():
    x = torch.randn(8, 20)
    torch.testing.assert_close(shift(x, 0), x)
    torch.testing.assert_close(shift(x, 1), rotate(x, 1))
    torch.testing.assert_close(shift(x, -1), rotate(x, -1))


def test_shift_interpolates_neighbours():
    x = torch.randn(8, 20)
    torch.testing.assert_close(shift(x, 0.25), 0.75 * x + 0.25 * rotate(x, 1))
    torch.testing.assert_close(shift(x, -0.5), 0.5 * x + 0.5 * rotate(x, -1))


def test_shift_draws_stay_in_range():
    aug = Augment(shift=0.5)
    aug.manual_seed(0)
    x = torch.tensor([0.0, 1, 0, 0, 0, 0, 0, 0])[:, None]  # a single 1 at ring position 1
    us = []
    for _ in range(300):
        y = aug(x)[:, 0]
        us.append((y * torch.tensor([0.0, 1, 2, 3, 4, 5, 6, -1])).sum().item() - 1)  # centre of mass - 1 = u
    assert max(abs(u) for u in us) <= 0.5 + 1e-6
    assert min(us) < -0.4 and max(us) > 0.4

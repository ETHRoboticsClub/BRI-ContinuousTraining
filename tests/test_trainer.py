import pytest
import torch
from torch import nn

from src.model.kernelized import Kernelized
from src.model.model import Model
from src.trainer import evaluate, param_groups


class ConstModel(nn.Module):
    """Returns a fixed prediction regardless of input."""

    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, x):
        return self.pred


def test_evaluate_metrics():
    y = torch.tensor([[0.0, 10.0], [2.0, 14.0]])
    pred = torch.tensor([[1.0, 10.0], [2.0, 10.0]])
    loader = [(torch.zeros(2, 1), y)]
    m = evaluate(ConstModel(pred), loader, "cpu")  # ConstModel returns pred regardless of input
    assert m["mae"] == pytest.approx((1 + 0 + 0 + 4) / 4)
    assert m["nmse"] == pytest.approx(1 - m["r2"])
    assert m["nmse"] == pytest.approx(((0.5 / 1.0) + (8.0 / 4.0)) / 2)


def dynamics(m):
    return {id(p) for k in m.modules() if isinstance(k, Kernelized) for p in (k.A, k.log_dt)}


def test_param_groups_default_is_one_group():
    m = Model(8, 15, 10)
    (g,) = param_groups(m, 1e-3, 1e-4)
    assert (g["lr"], g["weight_decay"]) == (1e-3, 1e-4)
    assert {id(p) for p in g["params"]} == {id(p) for p in m.parameters()}


@pytest.mark.parametrize("scale,decay_A", [(0.1, True), (1.0, False), (0.1, False)])
def test_param_groups_cover_every_parameter_once(scale, decay_A):
    m = Model(8, 15, 10)
    ids = [id(p) for g in param_groups(m, 1e-3, 1e-4, scale, decay_A) for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in m.parameters())


def test_ssm_lr_scale():
    m = Model(8, 15, 10)
    by_id = {id(p): g for g in param_groups(m, 1e-3, 1e-4, ssm_lr_scale=0.1) for p in g["params"]}
    for i, g in by_id.items():
        assert g["lr"] == pytest.approx(1e-4 if i in dynamics(m) else 1e-3) and g["weight_decay"] == 1e-4


def test_no_decay_on_A():
    m = Model(8, 15, 10)
    A = {id(k.A) for k in m.modules() if isinstance(k, Kernelized)}
    by_id = {id(p): g for g in param_groups(m, 1e-3, 1e-4, ssm_decay_A=False) for p in g["params"]}
    assert len(A) == 2
    for i, g in by_id.items():
        assert g["weight_decay"] == (0.0 if i in A else 1e-4) and g["lr"] == 1e-3


def test_ssm_lr_ratio_survives_cosine():
    m = Model(8, 15, 10)
    opt = torch.optim.AdamW(param_groups(m, 1e-3, 1e-4, ssm_lr_scale=0.1))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15)
    for _ in range(5):
        opt.step()
        sched.step()
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[0] / lrs[1] == pytest.approx(0.1)


def test_nonpositive_lr_scale_raises():
    with pytest.raises(ValueError):
        param_groups(Model(8, 15, 10), 1e-3, 1e-4, ssm_lr_scale=0)

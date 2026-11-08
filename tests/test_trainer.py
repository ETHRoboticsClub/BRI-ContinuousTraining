import pytest
import torch
from torch import nn

from src.trainer import evaluate


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

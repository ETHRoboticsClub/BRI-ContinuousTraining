import torch
from torch import nn

from src.model.kernelized import Kernelized, KernelizedLinear


def activation(name: str) -> nn.Module:
    """Fresh activation module: "tanh" or "relu"."""
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}, expected 'tanh' or 'relu'")


make_activation = activation  # SSMEncoder's `activation` argument shadows the function


class SSMEncoder(nn.Module):
    """Projection followed by two stacked Kernelized layers, returning the last tangent step.

    `downsample` s keeps every s-th tangent step of ssm1's output (MetaTF's Stride: x[:, s-1::s]), so ssm2 sees
    length / s steps and the last tangent step is kept. s must divide length and leave ssm2 at least 2 steps.
    """

    def __init__(
        self,
        in_dim: int,
        length: int,
        activation: str,
        dense: int = 256,
        units=(256, 128),
        dropout=0.1,
        downsample: int = 1,
    ):
        super().__init__()
        if downsample < 1 or length % downsample or length // downsample < 2:
            raise ValueError(f"downsample {downsample} must divide length {length} and leave at least 2 steps")
        self.downsample = downsample
        self.proj = KernelizedLinear(in_dim, dense)
        self.ssm1 = Kernelized(dense, units[0], length)
        self.ssm2 = Kernelized(units[0], units[1], length // downsample)
        self.act1, self.act2, self.act3 = (make_activation(activation) for _ in range(3))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = units[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tangent sequence [B, L, in_dim] -> [B, out_dim]."""
        h = self.act1(self.proj(x))
        h = self.ssm1(h)[:, self.downsample - 1 :: self.downsample]
        h = self.dropout(self.act2(h))
        h = self.act3(self.ssm2(h))
        return self.dropout(h[:, -1])

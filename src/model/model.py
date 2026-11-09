import torch
from torch import nn

from src.model.kernelized import KernelizedLinear
from src.model.ssm import SSMEncoder, make_activation
from src.model.tangent_space import TangentSpace, n_steps


class Model(nn.Module):
    """Temporal Riemannian Regressor (Colot et al., 2026), SSM variant.

    Tangent-space covariances -> projection -> two Kernelized SSM layers -> dense head. No layer has a bias.
    """

    def __init__(
        self,
        channels: int,
        targets: int,
        length: int,
        activation: str = "tanh",
        hidden: int = 64,
        downsample: int = 1,
    ) -> None:
        super().__init__()
        self.length = length
        self.tangent_space = TangentSpace(channels)
        self.encoder = SSMEncoder(self.tangent_space.out_dim, length, activation, downsample=downsample)
        self.head = nn.Sequential(
            KernelizedLinear(self.encoder.out_dim, hidden),
            make_activation(activation),
            KernelizedLinear(hidden, targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG training example [B, C, raw_length(length)] -> joint angles [B, targets]."""
        steps = n_steps(x.shape[-1])
        if steps != self.length:
            raise ValueError(f"expected {self.length} tangent steps, got {steps} from {x.shape[-1]} samples")
        return self.head(self.encoder(self.tangent_space(x)))

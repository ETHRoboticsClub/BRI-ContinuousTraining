import torch
from torch import nn

from src.model.gru import GRUEncoder
from src.model.tangent_space import TangentSpace


class Model(nn.Module):
    """Temporal Riemannian Regressor (Colot et al., 2026): tangent-space covariances -> GRUs -> dense head."""

    def __init__(self, channels, targets, hidden=64) -> None:
        super().__init__()
        self.tangent_space = TangentSpace(channels)
        self.gru = GRUEncoder(self.tangent_space.out_dim)
        self.head = nn.Sequential(nn.Linear(self.gru.out_dim, hidden), nn.Tanh(), nn.Linear(hidden, targets))

    def forward(self, x) -> torch.Tensor:
        """EMG window [B, C, W] -> joint angles [B, targets]."""
        return self.head(self.gru(self.tangent_space(x)))

import torch
from torch import nn

from src.model.gru import GRUEncoder
from src.model.tangent_space import TangentSpace


class Model(nn.Module):
    """Temporal Riemannian Regressor (Colot et al., 2026): tangent-space covariances -> GRUs -> dense head."""

    def __init__(self, channels: int = 8, targets: int = 15, hidden: int = 64, in_features: int | None = None) -> None:
        super().__init__()
        self.tangent_space = TangentSpace(channels)
        in_dim = in_features if in_features is not None else self.tangent_space.out_dim
        self.gru = GRUEncoder(in_dim)
        self.head = nn.Sequential(nn.Linear(self.gru.out_dim, hidden), nn.Tanh(), nn.Linear(hidden, targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG window [B, C, W] or tangent sequence [B, S, D] -> joint angles [B, targets]."""
        if x.ndim == 3 and x.shape[1] == self.tangent_space.channels and x.shape[2] >= self.tangent_space.sub_window:
            x = self.tangent_space(x)
        return self.head(self.gru(x))

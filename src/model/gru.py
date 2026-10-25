import torch
from torch import nn


class GRUEncoder(nn.Module):
    """Dense projection followed by two stacked GRUs, returning the last time step."""

    def __init__(self, in_dim, dense=256, units=(256, 128), dropout=0.1) -> None:
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(in_dim, dense), nn.Tanh())
        self.gru1 = nn.GRU(dense, units[0], batch_first=True)
        self.gru2 = nn.GRU(units[0], units[1], batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = units[1]

    def forward(self, x) -> torch.Tensor:
        """Sequence [B, S, in_dim] -> [B, out_dim]."""
        h, _ = self.gru1(self.dense(x))
        h, _ = self.gru2(self.dropout(h))
        return self.dropout(h[:, -1])

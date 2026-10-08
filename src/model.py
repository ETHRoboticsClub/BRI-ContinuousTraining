from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from src.probes import dead_units, probe, saturation, temporal_dynamics


class Model(nn.Module):
    """Example: causal temporal convolutions -> last time step -> MLP -> actuator activations in [0, 1].

    Replace it with your own. Input [B, channels, window], output [B, n_targets].
    """

    def __init__(self, in_channels, out_dim, hidden=32, layers=3, kernel_size=5, dropout=0.1):
        super().__init__()
        self.pad = kernel_size - 1
        self.convs = nn.ModuleList(
            nn.Conv1d(in_channels if i == 0 else hidden, hidden, kernel_size, dilation=2**i) for i in range(layers)
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, out_dim)

    @probe(
        "features",
        "activation_stats",
        "channel_activity",
        partial(dead_units, unit_axis=1),
        partial(temporal_dynamics, time_axis=-1),
    )
    def encode(self, x):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(F.pad(x, (self.pad * 2**i, 0))))
        return x

    @probe("embedding", "linear_probe", "target_cka", "effective_rank", "dead_units")
    @probe("embedding", "pca", every=5)
    def embed(self, x):
        return self.head(self.encode(x)[:, :, -1])

    @probe("output", saturation)
    def forward(self, x):
        return torch.sigmoid(self.out(self.embed(x)))

import math

import numpy as np
import torch
from torch import nn


def zoh_discretize(
    A: torch.Tensor, B: torch.Tensor, log_dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simplified ZOH: A [N, 2], B [in, N], log_dt [N] -> (A_hat, B_hat, dtA)."""
    A_c = torch.complex(-torch.exp(A[:, 0]), A[:, 1])
    dt = torch.exp(log_dt)
    dtA = dt * A_c
    B_hat = dt * B
    A_hat = torch.exp(dtA)
    return A_hat, B_hat, dtA


class Kernelized(nn.Module):
    """Diagonal complex SSM applied as a causal FFT convolution (port of MetaTF's Kernelized)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        num_coeffs: int = 32,
        repeat: int = 2,
        speed: float = 4,
        speed_range: float = 8,
    ) -> None:
        super().__init__()
        n = num_coeffs * repeat
        a = np.tile(
            np.stack([np.full(num_coeffs, math.log(0.5)), math.pi * np.arange(num_coeffs)], -1),
            (repeat, 1),
        )
        base = math.log(1 / (length - 1)) + math.log(speed)
        log_dt = np.repeat(np.linspace(base, base + math.log(speed_range), repeat), num_coeffs)
        self.A = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.B = nn.Parameter(torch.full((in_channels, n), 1 / math.sqrt(in_channels)))
        self.C = nn.Parameter(torch.randn(n, out_channels) * math.sqrt(2 / n))
        self.log_dt = nn.Parameter(torch.tensor(log_dt, dtype=torch.float32))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Sequence [B, T, in] -> [B, T, out]."""
        T = u.shape[1]
        _, B_hat, dtA = zoh_discretize(self.A, self.B, self.log_dt)
        x = torch.einsum("blc,cn->bnl", u, B_hat)
        lags = torch.arange(T, device=u.device, dtype=torch.float32)
        K = torch.exp(lags[None, :] * dtA[:, None]).real
        y = torch.fft.irfft(torch.fft.rfft(x, n=2 * T) * torch.fft.rfft(K, n=2 * T), n=2 * T)[..., :T]
        return torch.einsum("bnl,nd->bld", y, self.C)


class KernelizedLinear(nn.Module):
    """Memoryless linear projection: the Kernelized SSM with A_hat = 0, B_hat = I, D = 0.

    Each step computes ``u @ C``; the state size equals the input width. The MetaTF reference is
    ``StatefulProjection(use_bias=False)``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        a = math.sqrt(6 / (in_channels + out_channels))
        self.C = nn.Parameter(torch.empty(in_channels, out_channels).uniform_(-a, a))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """[..., in] -> [..., out]."""
        return u @ self.C

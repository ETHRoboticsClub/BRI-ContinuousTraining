import torch
from torch import nn


class TangentSpace(nn.Module):
    """Covariance of causal sub-windows, shrunk with OAS and mapped to the tangent space at the identity."""

    def __init__(self, channels, sub_window=60, step=20) -> None:
        super().__init__()
        self.sub_window, self.step = sub_window, step
        row, col = torch.triu_indices(channels, channels)
        self.register_buffer("row", row)
        self.register_buffer("col", col)
        self.register_buffer("weight", torch.where(row == col, 1.0, 2**0.5))
        self.out_dim = len(row)

    def covariance(self, x) -> torch.Tensor:
        """OAS-regularized covariance of x [..., C, T]."""
        c, t = x.shape[-2:]
        x = x - x.mean(-1, keepdim=True)
        s = x @ x.transpose(-1, -2) / (t - 1)
        mu = s.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
        tr_s2 = (s * s).sum((-2, -1), keepdim=True)
        num = (1 - 2 / c) * tr_s2 + (mu * c) ** 2
        den = (t + 1 - 2 / c) * (tr_s2 - (mu * c) ** 2 / c)
        rho = (num / (den + 1e-8)).clamp(0, 1)
        eye = torch.eye(c, device=x.device, dtype=x.dtype)
        return (1 - rho) * s + rho * mu * eye + 1e-5 * eye

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        """EMG window [B, C, W] -> sequence of tangent vectors [B, S, C*(C+1)/2], the last ending at sample W."""
        x = x[..., (x.shape[-1] - self.sub_window) % self.step :]
        subs = x.unfold(-1, self.sub_window, self.step).transpose(1, 2)
        evals, evecs = torch.linalg.eigh(self.covariance(subs))
        log = evecs @ torch.diag_embed(evals.clamp(min=1e-6).log()) @ evecs.transpose(-1, -2)
        return log[..., self.row, self.col] * self.weight

import numpy as np
import torch
from torch import nn


def ledoit_wolf_cov(x_centered: np.ndarray) -> np.ndarray:
    """Analytical Ledoit-Wolf shrinkage covariance for a zero-mean [C, T] matrix."""
    p, n = x_centered.shape
    if n <= 1:
        return np.eye(p, dtype=np.float32)
    xt = x_centered.T.astype(np.float64)
    emp_cov = (xt.T @ xt) / float(n)
    mu = float(np.trace(emp_cov)) / float(p)
    eye_p = np.eye(p, dtype=np.float64)
    delta = float(np.sum((emp_cov - mu * eye_p) ** 2)) / float(p)
    if delta <= 0.0:
        return emp_cov.astype(np.float32)
    x2 = xt**2
    beta_0 = (float(np.sum(x2.T @ x2)) / float(p * n * n)) - (float(np.sum(emp_cov**2)) / float(p * n))
    shrinkage = min(delta, max(0.0, beta_0)) / delta
    return ((1.0 - shrinkage) * emp_cov + shrinkage * mu * eye_p).astype(np.float32)


def riemannian_whitening(covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Riemannian mean covariance and inverse square-root whitening matrix C_mean^{-1/2}."""
    try:
        try:
            from pyriemann.geometry.mean import mean_covariance
        except ImportError:
            from pyriemann.utils.mean import mean_covariance

        sub_sample = covs[:: max(1, len(covs) // 500)]
        mean_cov = mean_covariance(sub_sample.astype(np.float64), metric="riemann").astype(np.float32)
    except Exception:
        mean_cov = np.mean(covs, axis=0).astype(np.float32)

    evals, evecs = np.linalg.eigh(mean_cov)
    evals = np.maximum(evals, 1e-5)
    inv_sqrt = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T).astype(np.float32)
    return mean_cov, inv_sqrt


class TangentSpace(nn.Module):
    """Covariance of causal sub-windows, shrunk with OAS and mapped to the tangent space at the identity."""

    def __init__(self, channels: int = 8, sub_window: int = 60, step: int = 20) -> None:
        super().__init__()
        self.channels = channels
        self.sub_window, self.step = sub_window, step
        row, col = torch.triu_indices(channels, channels)
        self.register_buffer("row", row)
        self.register_buffer("col", col)
        self.register_buffer("weight", torch.where(row == col, 1.0, 2**0.5))
        self.out_dim = len(row)

    def covariance(self, x: torch.Tensor) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG window [B, C, W] -> sequence of tangent vectors [B, S, C*(C+1)/2], the last ending at sample W."""
        x = x[..., (x.shape[-1] - self.sub_window) % self.step :]
        subs = x.unfold(-1, self.sub_window, self.step).transpose(1, 2)
        evals, evecs = torch.linalg.eigh(self.covariance(subs))
        log = evecs @ torch.diag_embed(evals.clamp(min=1e-6).log()) @ evecs.transpose(-1, -2)
        return log[..., self.row, self.col] * self.weight

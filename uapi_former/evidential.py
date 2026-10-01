"""Normal-Inverse-Gamma (NIG) evidential head and loss.

Implements the evidential regression framework from:
    Amini et al., "Deep Evidential Regression", NeurIPS 2020.
    https://arxiv.org/abs/1910.02600

The NIG prior over a Gaussian likelihood gives four output parameters per
target: (gamma, v, alpha, beta), where
    predictive mean      = gamma
    aleatoric variance   ~ beta / (alpha - 1)
    epistemic variance   ~ beta / (v * (alpha - 1))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialNIGHead(nn.Module):
    """Linear layer that maps `in_features` -> 4 NIG parameters per target.

    Output order per target: (gamma, v, alpha, beta).
    All positivity constraints are enforced via softplus.
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features * 4)

    def forward(self, x: torch.Tensor):
        params = self.fc(x).view(-1, self.out_features, 4)
        gamma = params[..., 0]
        v     = F.softplus(params[..., 1]) + 1e-6          # > 0
        alpha = F.softplus(params[..., 2]) + 1.0 + 1e-6    # > 1  (required for finite variance)
        beta  = F.softplus(params[..., 3]) + 1e-6          # > 0
        if self.out_features == 1:
            return gamma.squeeze(-1), v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)
        return gamma, v, alpha, beta


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def nig_predictive_variance(v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Predictive variance: beta * (1 + v) / (v * (alpha - 1))."""
    return beta * (1.0 + v) / (v * (alpha - 1.0 + eps) + eps)


def nig_aleatoric_variance(alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Aleatoric (irreducible) variance: beta / (alpha - 1)."""
    return beta / (alpha - 1.0 + eps)


def nig_epistemic_variance(v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Epistemic (model) variance: beta / (v * (alpha - 1))."""
    return beta / (v * (alpha - 1.0 + eps) + eps)


# ---------------------------------------------------------------------------
# Full Amini 2020 NIG loss
# ---------------------------------------------------------------------------

def nig_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    coeff: float = 1e-2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Amini et al. NeurIPS 2020 evidential NIG loss.

    L = L_NLL + coeff * L_reg

    L_NLL (marginal log-likelihood of t under the NIG predictive):
        = 0.5 * log(pi / v)
          - alpha * log(2*beta*(1 + v))
          + (alpha + 0.5) * log((y - gamma)^2 * v + 2*beta*(1 + v))
          + log(Gamma(alpha) / Gamma(alpha + 0.5))

    L_reg (evidence regulariser, penalises high evidence on wrong predictions):
        = |y - gamma| * (2*v + alpha)

    Args:
        y:      ground truth, shape (B,)
        gamma, v, alpha, beta: NIG parameters, shape (B,)
        coeff:  regularisation weight (lambda in the paper)
    """
    y = y.view_as(gamma)

    two_beta = 2.0 * beta
    one_plus_v = 1.0 + v

    # log-likelihood terms
    t1 = 0.5 * (torch.log(torch.tensor(torch.pi, device=y.device, dtype=y.dtype)) - torch.log(v + eps))
    t2 = -alpha * torch.log(two_beta * one_plus_v + eps)
    err_sq = (y - gamma) ** 2
    t3 = (alpha + 0.5) * torch.log(err_sq * v + two_beta * one_plus_v + eps)
    t4 = torch.lgamma(alpha + 0.5) - torch.lgamma(alpha + eps)

    nll = t1 + t2 + t3 - t4

    # evidence regulariser
    reg = torch.abs(y - gamma) * (2.0 * v + alpha)

    return (nll + coeff * reg).mean()

"""Evaluation metrics used by the UAPI-Former prototype."""
from typing import Sequence, Dict, Tuple
import numpy as np
import torch

from .evidential import nig_predictive_variance


def rmse(preds, targets) -> float:
    preds = torch.as_tensor(preds).float()
    targets = torch.as_tensor(targets).float()
    return float(torch.sqrt(torch.mean((preds - targets) ** 2)).cpu().item())


def mae(preds, targets) -> float:
    preds = torch.as_tensor(preds).float()
    targets = torch.as_tensor(targets).float()
    return float(torch.mean(torch.abs(preds - targets)).cpu().item())


def ece_from_mean_var(mu, var, y, probs: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Tuple[float, Dict[float, float]]:
    """Compute a simple regression ECE across several central-coverage levels.

    - `mu`, `var`, `y` may be numpy arrays or torch tensors of the same shape.
    - `probs` are central coverage probabilities (e.g. 0.95 for 95% CI).

    Returns: (ece, coverage_dict) where coverage_dict maps nominal->observed coverage.
    """
    mu = torch.as_tensor(mu).float()
    var = torch.as_tensor(var).float()
    y = torch.as_tensor(y).float()

    std = torch.sqrt(var.clamp(min=1e-12))
    normal = torch.distributions.Normal(0.0, 1.0)

    errors = []
    coverages: Dict[float, float] = {}
    for p in probs:
        z = normal.icdf(torch.tensor((1.0 + p) / 2.0))
        lower = mu - z * std
        upper = mu + z * std
        covered = ((y >= lower) & (y <= upper)).float().mean().cpu().item()
        coverages[p] = covered
        errors.append(abs(covered - p))

    ece = float(np.mean(errors))
    return ece, coverages


def ece_from_nig(gamma, v, alpha, beta, y, probs: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Tuple[float, Dict[float, float]]:
    var = nig_predictive_variance(v, alpha, beta)
    return ece_from_mean_var(gamma, var, y, probs)

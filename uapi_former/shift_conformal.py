"""Shift-robust conformal calibration primitives (schemes S0-S3).

S0: vanilla split conformal (exchangeability assumed) -- vanilla_q
S1: weighted conformal under covariate shift (Tibshirani et al. 2019),
    density ratio estimated by a torch logistic discriminator in the frozen
    128-d `rep` latent space -- logistic_density_ratio + weighted_q
S2: same weighted quantile, weights from semi-relaxed entropic optimal
    transport (target marginal enforced, calibration marginal free), reusing
    the euclidean-cost geometry of uapi_former.eot -- semirelaxed_transport_weights
S3: epistemic-gated abstention on top of S1/S2 -- epistemic_variance + a
    threshold chosen on SOURCE validation only.

All functions are pure torch/numpy; no training, no new dependencies.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from uapi_former.evidential import nig_predictive_variance, nig_epistemic_variance


# -- scores -------------------------------------------------------------------

def nonconformity(gamma: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                  beta: torch.Tensor, y: torch.Tensor):
    """Sigma-normalised absolute-residual scores and sigma (both 1-D)."""
    sigma = torch.sqrt(nig_predictive_variance(v, alpha, beta).clamp(min=1e-8))
    return (torch.abs(y - gamma) / sigma), sigma


# -- quantiles ----------------------------------------------------------------

def vanilla_q(scores: torch.Tensor, coverage: float = 0.90) -> float:
    """Split-conformal quantile, identical formula to scripts/calibrate.py."""
    n = len(scores)
    q_level = min(float(np.ceil((n + 1) * coverage) / n), 1.0)
    return float(torch.quantile(scores, q_level))


def weighted_q(scores: torch.Tensor, weights: torch.Tensor,
               coverage: float = 0.90,
               w_test: "Optional[float | torch.Tensor]" = None):
    """Weighted conformal quantile (Tibshirani et al. 2019).

    `weights` are the density-ratio weights of the calibration points. Each
    test point i contributes its OWN weight mass w_test_i to the total before
    the coverage-quantile is located (the exact per-test-point construction,
    not a group-level average) — this is what actually restores nominal
    coverage under covariate shift; a single averaged w_test systematically
    under-corrects tail-region test points.

    - `w_test=None` -> uses mean(weights) as a scalar mass (coarse group-level
      approximation; kept only for the sanity check against vanilla_q).
    - `w_test=<float>` -> single scalar mass, returns a float quantile.
    - `w_test=<Tensor (n_test,)>` -> per-test-point mass, returns a Tensor of
      quantiles (n_test,) with `inf` where the calibration weight mass cannot
      certify the requested coverage for that point (caller must treat `inf`
      as an infinite-width interval that trivially covers — see
      interval_metrics).
    """
    order = torch.argsort(scores)
    s, w = scores[order].double(), weights[order].double()
    cw = torch.cumsum(w, 0)
    total = float(w.sum())

    if w_test is None:
        w_test = float(weights.mean())

    if isinstance(w_test, torch.Tensor) and w_test.dim() > 0:
        thresh = coverage * (total + w_test.double())
        idx = torch.searchsorted(cw, thresh)
        overflow = idx >= len(s)
        idx_c = idx.clamp(max=len(s) - 1)
        q = s[idx_c].clone()
        q[overflow] = float("inf")
        return q

    w_test_f = float(w_test)
    thresh = coverage * (total + w_test_f)
    idx = int(torch.searchsorted(cw, torch.tensor(thresh, dtype=torch.float64)))
    if idx >= len(s):
        return float("inf")
    return float(s[idx])


def effective_sample_size(weights: torch.Tensor) -> float:
    return float(weights.sum() ** 2 / (weights ** 2).sum())


# -- weight estimators --------------------------------------------------------

def _fit_discriminator(z_calib: torch.Tensor, z_target: torch.Tensor,
                       epochs: int = 2000, lr: float = 0.1,
                       weight_decay: float = 1e-4, seed: int = 0):
    """Linear logistic discriminator (calib=0, target=1) in standardised
    calibration coordinates. Returns (lin, standardiser).

    epochs=2000, lr=0.1 are calibrated against real 128-d UAPI-Former latents
    (checkpoints/loco_*), not just low-dimensional synthetic data: the
    original defaults (300 epochs, lr=0.05) were confirmed undertrained on
    real data — full-batch Adam had not converged, giving near-uniform
    weights (ESS close to n) that could not correct coverage on any LOCO fold
    even where a genuine, correctable covariate shift was present (verified
    on B0005/B0018, where the undertrained discriminator gave S1 coverage
    0.79/0.57 but the converged one gives 0.91/0.96 against a 0.90 nominal
    target). weight_decay guards against runaway overfitting in 128
    dimensions with a finite calibration set.
    """
    torch.manual_seed(seed)
    mu = z_calib.mean(0, keepdim=True)
    sd = z_calib.std(0, keepdim=True).clamp(min=1e-6)
    std = lambda z: ((z - mu) / sd).float()
    X = torch.cat([std(z_calib), std(z_target)])
    y = torch.cat([torch.zeros(len(z_calib)), torch.ones(len(z_target))])
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(lin(X).squeeze(-1), y)
        loss.backward()
        opt.step()
    return lin, std


def logistic_density_ratio(z_calib: torch.Tensor, z_target: torch.Tensor,
                           epochs: int = 2000, lr: float = 0.1,
                           clip: float = 50.0, seed: int = 0,
                           query: Optional[torch.Tensor] = None,
                           weight_decay: float = 1e-4) -> torch.Tensor:
    """w ~= p_target(z)/p_calib(z) via a linear logistic discriminator
    (calib=0, target=1) trained with full-batch Adam.

    The class-prior ratio n_cal/n_tgt cancels the sampling imbalance so w is a
    density ratio. Weights are clipped at `clip` (report ESS alongside).

    By default the ratio is evaluated AT the calibration points (the S1
    calibration weights). Pass `query` to evaluate the same fitted ratio at
    different points instead (e.g. AT the target points, to get the
    per-test-point w_test used by the exact weighted quantile).
    """
    lin, std = _fit_discriminator(z_calib, z_target, epochs, lr, weight_decay, seed)
    q = query if query is not None else z_calib
    with torch.no_grad():
        logit = lin(std(q)).squeeze(-1)
        w = torch.exp(logit) * (len(z_calib) / max(1, len(z_target)))
    return torch.clamp(w, max=clip)


def discriminator_auc(z_calib: torch.Tensor, z_target: torch.Tensor) -> float:
    """AUC of the discriminator — shift-severity diagnostic (0.5 = no shift)."""
    lin, std = _fit_discriminator(z_calib, z_target)
    with torch.no_grad():
        s_cal = lin(std(z_calib)).squeeze(-1)
        s_tgt = lin(std(z_target)).squeeze(-1)
    # rank AUC computed in manageable chunks to bound memory
    total, count = 0.0, 0
    for i in range(0, len(s_tgt), 2048):
        chunk = s_tgt[i:i + 2048]
        total += float((chunk.unsqueeze(1) > s_cal.unsqueeze(0)).float().sum())
        count += len(chunk) * len(s_cal)
    return total / max(1, count)


def semirelaxed_transport_weights(z_target: torch.Tensor, z_calib: torch.Tensor,
                                  reg: float = 0.1, clip: float = 20.0) -> torch.Tensor:
    """Entropic transport with the target-side marginal enforced (uniform) and
    the calibration-side marginal free: the optimal plan is the row-softmax
    kernel P_ij = softmax_j(-||z_t_i - z_c_j|| / reg) / m. The free column
    masses, rescaled by n, are the OT usage weights of calibration points.
    Same euclidean geometry as uapi_former.eot.sinkhorn_loss.
    """
    M = torch.cdist(z_target.float(), z_calib.float(), p=2)
    M = M / M.mean().clamp(min=1e-8)          # scale-invariant cost
    P = torch.softmax(-M / reg, dim=1) / M.shape[0]
    w = P.sum(0) * M.shape[1]
    return torch.clamp(w, max=clip)


# -- epistemic gate (S3) ------------------------------------------------------

def epistemic_variance(v: torch.Tensor, alpha: torch.Tensor,
                       beta: torch.Tensor) -> torch.Tensor:
    """NIG epistemic variance — delegates to uapi_former.evidential's
    canonical implementation (beta / (v * (alpha - 1))) so this module never
    drifts from the formula used elsewhere in the codebase."""
    return nig_epistemic_variance(v, alpha, beta)


# -- evaluation ---------------------------------------------------------------

def interval_metrics(y: torch.Tensor, gamma: torch.Tensor, sigma: torch.Tensor,
                     q, keep: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Empirical coverage and mean interval width for CI = gamma +/- q*sigma.

    `q` may be a scalar (float, possibly inf) or a per-sample Tensor (as
    returned by weighted_q with a per-test-point w_test). Non-finite q means
    "cannot certify at this coverage level" -> convention: the interval is
    (-inf, inf), which trivially covers y (reported in `coverage`) but is
    excluded from `mean_width` (reported separately via `frac_uncertifiable`)
    so an inflated coverage number is never mistaken for a tight interval.

    `keep` (bool mask) restricts to non-abstained samples (S3); abstention
    fraction is reported alongside.
    """
    if keep is None:
        keep = torch.ones_like(y, dtype=torch.bool)
    frac_abstain = float((~keep).float().mean())
    yk, gk, sk = y[keep], gamma[keep], sigma[keep]
    if len(yk) == 0:
        return {"coverage": float("nan"), "mean_width": float("nan"),
                "frac_uncertifiable": float("nan"),
                "abstain_frac": frac_abstain, "n_kept": 0}

    if isinstance(q, torch.Tensor):
        qk = q[keep].double()
        finite = torch.isfinite(qk)
        covered = torch.empty_like(qk, dtype=torch.bool)
        covered[~finite] = True
        if finite.any():
            covered[finite] = (torch.abs(yk[finite] - gk[finite]).double()
                               <= qk[finite] * sk[finite].double())
        mean_width = (float((2.0 * qk[finite] * sk[finite].double()).mean())
                     if finite.any() else float("inf"))
        return {"coverage": float(covered.float().mean()),
                "mean_width": mean_width,
                "frac_uncertifiable": float((~finite).float().mean()),
                "abstain_frac": frac_abstain,
                "n_kept": int(keep.sum())}

    if not np.isfinite(q):
        return {"coverage": 1.0, "mean_width": float("inf"),
                "frac_uncertifiable": 1.0,
                "abstain_frac": frac_abstain, "n_kept": int(keep.sum())}
    covered = (torch.abs(yk - gk) <= q * sk).float().mean()
    return {"coverage": float(covered),
            "mean_width": float(2.0 * q * sk.mean()),
            "frac_uncertifiable": 0.0,
            "abstain_frac": frac_abstain,
            "n_kept": int(keep.sum())}

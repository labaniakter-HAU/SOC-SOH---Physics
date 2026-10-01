"""Electrochemical Optimal Transport (EOT) primitives.

Weight-frozen cross-chemistry adaptation: a lightweight residual transducer
T_psi warps target-chemistry pooled latents (`rep`, B x d_model) onto the
NASA source latent manifold by entropic Sinkhorn OT, regularised by a dQ/dV
phase-preserving term so the warp cannot destroy the target cell's
incremental-capacity (phase-transition) signature.

The UAPI-Former backbone is never modified or retrained. `encode_to_rep` and
`decode_from_rep` replicate the two halves of `UAPIFormer.forward` around the
`rep` tap point (model.py:344) using the model's existing frozen submodules.

Smoke test:
    python -m uapi_former.eot --smoke
"""
from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Transducer ───────────────────────────────────────────────────────────────

class OTTransducer(nn.Module):
    """Residual MLP R^d -> R^d, initialised at identity (T(z) = z at init)."""

    def __init__(self, d_model: int = 128, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        # Zero-init last layer => residual starts as identity (stability anchor).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def delta(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(z)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.delta(z)


# ── Frozen backbone encode / decode around the `rep` tap ─────────────────────

def encode_to_rep(model, x: torch.Tensor,
                  v_spme: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate UAPIFormer.forward up to the pooled `rep` (model.py:326-344).

    Returns (rep, ic_feat): rep is (B, d_model); ic_feat is (B, d_model//4),
    the projected IC vector consumed by the SOH head.
    """
    ic_feat = model.ic_extractor(x)
    residual_mag = torch.abs(x[..., 0] - v_spme) if v_spme is not None else None
    chem_token = model.dct(x).unsqueeze(1)
    h = model.eite(x, v_spme)
    h = model.pos_enc(h)
    h = torch.cat([chem_token, h], dim=1)
    h = model.encoder(h)
    rep = model.prap(h[:, 1:, :], residual_mag)
    return rep, ic_feat


def decode_from_rep(model, rep: torch.Tensor, ic_feat: torch.Tensor):
    """Replicate UAPIFormer.forward from `rep` onward (model.py:347-355).

    Returns (soc_out, soh_out) NIG tuples; prediction is element [0] (gamma).
    """
    soc_rep, soh_rep = model.dtag(rep)
    soc_rep, soh_rep = model.ctba(soc_rep, soh_rep)
    soh_rep_full = torch.cat([soh_rep, ic_feat], dim=-1)
    return model.soc_head(soc_rep), model.soh_head(soh_rep_full)


# ── dQ/dV soft-histogram (mirrors ICFeatureExtractor pre-projection) ─────────

def ic_histogram(x: torch.Tensor, n_bins: int = 20) -> torch.Tensor:
    """Differentiable dQ/dV soft-histogram over voltage bins, (B, n_bins).

    Mirrors ICFeatureExtractor.forward (model.py:157-181) BEFORE the learned
    projection, so it is the raw phase-transition signature used by R_phase.
    """
    V = x[..., 0]
    I = x[..., 1]
    dV = V[:, 1:] - V[:, :-1]
    dQ = -I[:, :-1]
    ic = dQ / (dV.abs().clamp(min=1e-6)) * dV.sign()
    ic = ic.clamp(-50.0, 50.0)
    V_mid = V[:, :-1]
    v_min = V_mid.min(dim=-1, keepdim=True).values
    v_max = V_mid.max(dim=-1, keepdim=True).values
    v_norm = (V_mid - v_min) / (v_max - v_min + 1e-6) * (n_bins - 1)
    bins = torch.arange(n_bins, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (v_norm.unsqueeze(-1) - bins) ** 2)
    ic_hist = (kernel * ic.unsqueeze(-1)).sum(dim=1)
    ic_hist = ic_hist / (kernel.sum(dim=1) + 1e-6)
    return ic_hist


# ── Sinkhorn alignment loss (entropic W1) ────────────────────────────────────

def sinkhorn_loss(z_a: torch.Tensor, z_b: torch.Tensor, reg: float = 0.1,
                  n_iters: int = 100) -> torch.Tensor:
    """Entropic-regularised Wasserstein-1 between two latent minibatches.

    Self-contained, log-domain-stabilised Sinkhorn (no external OT dependency:
    POT ships no wheel for Python 3.14). Uniform marginals; euclidean ground
    cost (W1). Fully differentiable w.r.t. z_a (the warped target).
    """
    M = torch.cdist(z_a, z_b, p=2)                      # (n, m) euclidean cost
    n, m = M.shape
    log_a = torch.full((n,), -float(np.log(n)), device=M.device, dtype=M.dtype)
    log_b = torch.full((m,), -float(np.log(m)), device=M.device, dtype=M.dtype)
    Mr = M / reg
    f = torch.zeros(n, device=M.device, dtype=M.dtype)
    g = torch.zeros(m, device=M.device, dtype=M.dtype)
    for _ in range(n_iters):
        f = log_a - torch.logsumexp(-Mr + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(-Mr + f.unsqueeze(1), dim=0)
    P = torch.exp(f.unsqueeze(1) + g.unsqueeze(0) - Mr)  # transport plan
    return (P * M).sum()


# ── IC probe: linear map source rep -> source IC histogram (frozen) ──────────

def fit_ic_probe(rep_src: torch.Tensor, ic_src: torch.Tensor,
                 ridge: float = 1e-2) -> torch.Tensor:
    """Closed-form ridge regression W: [rep | 1] -> ic_hist on the SOURCE domain.

    Returns W of shape (d_model + 1, n_bins). Used by R_phase to test whether a
    warped target latent still decodes to the target's own dQ/dV signature.
    """
    n, d = rep_src.shape
    ones = torch.ones(n, 1, device=rep_src.device, dtype=rep_src.dtype)
    R = torch.cat([rep_src, ones], dim=1)              # (n, d+1)
    A = R.t() @ R + ridge * torch.eye(d + 1, device=R.device, dtype=R.dtype)
    W = torch.linalg.solve(A, R.t() @ ic_src)          # (d+1, n_bins)
    return W


def apply_ic_probe(W: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(rep.shape[0], 1, device=rep.device, dtype=rep.dtype)
    return torch.cat([rep, ones], dim=1) @ W


# ── Phase-preserving regularizer variants ────────────────────────────────────

def _soft_argmax(h: torch.Tensor) -> torch.Tensor:
    """Soft peak location along the bin axis, (B,)."""
    n_bins = h.shape[-1]
    idx = torch.arange(n_bins, device=h.device, dtype=h.dtype)
    w = torch.softmax(h, dim=-1)
    return (w * idx).sum(dim=-1)


def phase_regularizer(pred_ic: torch.Tensor, true_ic: torch.Tensor,
                      variant: str) -> torch.Tensor:
    """R_phase between probe-predicted IC (from warped latent) and true target IC.

    variant in {"none", "grad", "peak", "dist"}.
    """
    if variant == "none":
        return torch.zeros((), device=pred_ic.device, dtype=pred_ic.dtype)
    if variant == "grad":  # match gradient (shape) along the voltage-bin axis
        dp = pred_ic[:, 1:] - pred_ic[:, :-1]
        dt = true_ic[:, 1:] - true_ic[:, :-1]
        return F.mse_loss(dp, dt)
    if variant == "peak":  # match soft peak location
        return F.mse_loss(_soft_argmax(pred_ic), _soft_argmax(true_ic))
    if variant == "dist":  # symmetric KL between normalised IC distributions
        p = torch.softmax(pred_ic, dim=-1)
        q = torch.softmax(true_ic, dim=-1)
        eps = 1e-8
        kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)
        kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(-1)
        return 0.5 * (kl_pq + kl_qp).mean()
    raise ValueError(f"unknown phase variant: {variant}")


PHASE_VARIANTS = ["none", "grad", "peak", "dist"]


# ── Self-contained smoke test ────────────────────────────────────────────────

def _smoke() -> None:
    torch.manual_seed(0)
    d, n, nb = 128, 64, 20
    rep_src = torch.randn(n, d)
    ic_src = torch.randn(n, nb)
    rep_tgt = torch.randn(n, d) + 0.5
    ic_tgt = torch.randn(n, nb)

    T = OTTransducer(d_model=d)
    assert torch.allclose(T(rep_tgt), rep_tgt, atol=1e-6), "transducer must init at identity"

    W = fit_ic_probe(rep_src, ic_src)
    assert W.shape == (d + 1, nb)

    opt = torch.optim.Adam(T.parameters(), lr=1e-3)
    for variant in PHASE_VARIANTS:
        opt.zero_grad()
        z = T(rep_tgt)
        loss = sinkhorn_loss(z, rep_src, reg=0.1)
        pred_ic = apply_ic_probe(W, z)
        loss = loss + 1.0 * phase_regularizer(pred_ic, ic_tgt, variant)
        loss = loss + 0.1 * T.delta(rep_tgt).pow(2).mean()  # identity anchor
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), f"non-finite loss for variant {variant}"
        print(f"  variant={variant:5s}  loss={float(loss):.4f}")
    print("EOT smoke test OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        _smoke()

"""Unit tests for shift-robust conformal primitives.

Run: .venv/Scripts/python.exe -m pytest tests/test_shift_conformal.py -v
(If pytest is unavailable in the venv, run the module directly:
 .venv/Scripts/python.exe tests/test_shift_conformal.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from uapi_former.shift_conformal import (
    vanilla_q, weighted_q, effective_sample_size,
    logistic_density_ratio, semirelaxed_transport_weights,
    epistemic_variance, interval_metrics,
)


def test_weighted_q_uniform_weights_matches_vanilla():
    torch.manual_seed(0)
    scores = torch.rand(500)
    w = torch.ones(500)
    assert abs(weighted_q(scores, w, 0.90) - vanilla_q(scores, 0.90)) < 0.02


def test_weighted_q_monotone_in_coverage():
    torch.manual_seed(0)
    scores = torch.rand(500); w = torch.rand(500) + 0.5
    assert weighted_q(scores, w, 0.95) >= weighted_q(scores, w, 0.90)


def test_effective_sample_size():
    assert abs(effective_sample_size(torch.ones(100)) - 100.0) < 1e-4
    w = torch.zeros(100); w[0] = 1.0
    assert abs(effective_sample_size(w) - 1.0) < 1e-4


def test_logistic_density_ratio_detects_shift():
    torch.manual_seed(0)
    z_cal = torch.randn(400, 8)
    z_tgt = torch.randn(400, 8) + torch.tensor([2.0] + [0.0] * 7)
    w = logistic_density_ratio(z_cal, z_tgt)
    # calib points nearer the target cloud (larger dim-0) must get larger weight
    hi = w[z_cal[:, 0] > 1.0].mean(); lo = w[z_cal[:, 0] < -1.0].mean()
    assert hi > 2.0 * lo


def test_transport_weights_detect_shift():
    torch.manual_seed(0)
    z_cal = torch.randn(300, 8)
    z_tgt = torch.randn(300, 8) + torch.tensor([2.0] + [0.0] * 7)
    w = semirelaxed_transport_weights(z_tgt, z_cal)
    assert w.shape == (300,)
    hi = w[z_cal[:, 0] > 1.0].mean(); lo = w[z_cal[:, 0] < -1.0].mean()
    assert hi > 2.0 * lo


def test_weighted_conformal_restores_coverage_under_covariate_shift():
    """End-to-end synthetic check: y = x + heteroscedastic noise, target
    distribution shifted in x. Vanilla CP undercovers; the exact
    per-test-point weighted CP must land within 3 points of nominal.

    Uses the textbook Tibshirani et al. 2019 construction: each test point
    contributes its OWN density-ratio weight mass before locating the
    coverage-quantile (not a single averaged mass, which under-corrects
    tail-region test points). Points where the calibration weight mass
    cannot certify the requested coverage get an infinite-width interval
    (trivially covers) -- see interval_metrics' frac_uncertifiable.
    """
    rng = np.random.default_rng(1)
    def sample(mu, n):
        x = rng.normal(mu, 1.0, n)
        y = x + rng.normal(0, 0.5 + 0.5 * np.abs(x), n)  # noise grows with |x|
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    x_cal, y_cal = sample(0.0, 2000)
    x_tst, y_tst = sample(2.5, 2000)          # covariate shift -> larger noise
    # model: yhat = x, sigma = const 1.0 (misspecified on purpose)
    s_cal = (y_cal - x_cal).abs()
    s_tst = (y_tst - x_tst).abs()
    q0 = vanilla_q(s_cal, 0.90)
    cov0 = float((s_tst <= q0).float().mean())

    # This synthetic shift is deliberately severe (true density ratio reaches
    # into the hundreds), so the library's real-data-oriented default clip=20
    # would truncate legitimate tail mass here; clip=100 is specific to this
    # synthetic scenario, not a change to the library default used on the
    # noisier real battery latents (which stays at clip=20 per the spec).
    w_cal = logistic_density_ratio(x_cal.unsqueeze(1), x_tst.unsqueeze(1), clip=100.0)
    w_tst = logistic_density_ratio(x_cal.unsqueeze(1), x_tst.unsqueeze(1),
                                   query=x_tst.unsqueeze(1), clip=100.0)
    q1 = weighted_q(s_cal, w_cal, 0.90, w_test=w_tst)
    finite = torch.isfinite(q1)
    covered = torch.empty_like(q1, dtype=torch.bool)
    covered[~finite] = True
    covered[finite] = s_tst.double()[finite] <= q1.double()[finite]
    cov1 = float(covered.float().mean())

    assert cov0 < 0.87                      # vanilla demonstrably fails
    assert abs(cov1 - 0.90) < 0.03          # weighted restores


def test_epistemic_variance_positive_and_decreasing_in_nu():
    v = torch.tensor([1.0, 10.0]); a = torch.tensor([2.0, 2.0]); b = torch.tensor([1.0, 1.0])
    ev = epistemic_variance(v, a, b)
    assert (ev > 0).all() and ev[0] > ev[1]


def test_interval_metrics():
    y = torch.tensor([0.0, 0.0, 0.0, 10.0])
    gamma = torch.zeros(4); sigma = torch.ones(4)
    m = interval_metrics(y, gamma, sigma, q=1.0)
    assert abs(m["coverage"] - 0.75) < 1e-6
    assert abs(m["mean_width"] - 2.0) < 1e-6
    assert m["frac_uncertifiable"] == 0.0


def test_interval_metrics_scalar_inf_q_trivially_covers():
    y = torch.tensor([0.0, 5.0]); gamma = torch.zeros(2); sigma = torch.ones(2)
    m = interval_metrics(y, gamma, sigma, q=float("inf"))
    assert m["coverage"] == 1.0
    assert m["mean_width"] == float("inf")
    assert m["frac_uncertifiable"] == 1.0


def test_interval_metrics_tensor_q_mixed_finite_and_inf():
    y = torch.tensor([0.0, 0.0, 100.0])
    gamma = torch.zeros(3); sigma = torch.ones(3)
    q = torch.tensor([1.0, 1.0, float("inf")])
    m = interval_metrics(y, gamma, sigma, q=q)
    # first two covered by q=1 interval, third trivially covered (inf q)
    assert abs(m["coverage"] - 1.0) < 1e-6
    assert abs(m["mean_width"] - 2.0) < 1e-6   # only the two finite q's average in
    assert abs(m["frac_uncertifiable"] - (1 / 3)) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")

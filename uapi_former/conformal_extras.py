"""Pure helpers for the weakness-removal analyses (spec 2026-09-24, SP1).

Kept free of I/O so every rule used in the new analyses is unit-tested in
tests/test_conformal_extras.py.
"""
import numpy as np
import torch

from uapi_former import shift_conformal as sc

ACTIVE_EPS = 1e-6      # same threshold as scripts/revision_rest_windows.py (label_zero = soc <= 1e-6)
COV = 0.90
TASKS = {"soc": ("sg", "sv", "sa", "sb", "soc"), "soh": ("hg", "hv", "ha", "hb", "soh")}


def active_mask(soc):
    """Windows whose SOC label is positive, i.e. not end-of-discharge or rest windows."""
    return torch.as_tensor(soc) > ACTIVE_EPS


def subset(d, mask):
    """Apply one boolean window mask to every field of a cache dict."""
    m = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    return {k: v[m] for k, v in d.items()}


def restrict_active(cal, tgt, *tgt_arrays):
    """Calibration and target caches restricted to active windows (SOC label > 0), for
    calibration AND evaluation (spec 2026-09-24 §4.1). Extra per-target-window arrays
    (cycle ids, window starts) are filtered with the target mask."""
    tm = active_mask(tgt["soc"])
    extra = [np.asarray(x)[tm.numpy()] for x in tgt_arrays]
    return (subset(cal, active_mask(cal["soc"])), subset(tgt, tm), *extra)


def scores(d, task, absolute=False):
    """(scores, sigma, y, gamma) for 'soc' or 'soh'.

    absolute=False: sigma-normalised NIG score (the v12-NIG lineage).
    absolute=True : plain absolute residual, sigma = 1 (MSE-only checkpoints).
    """
    g, v, a, b, y = (d[k] for k in TASKS[task])
    if absolute:
        return torch.abs(y - g), torch.ones_like(g), y, g
    s, sigma = sc.nonconformity(g, v, a, b, y)
    return s, sigma, y, g


def summarize(y, g, sigma, q):
    """Coverage, refusal, accepted coverage, finite-and-correct fraction, mean width (label percentage points)."""
    m = sc.interval_metrics(y, g, sigma, q)
    r = m.get("frac_uncertifiable", 0.0) or 0.0
    w = m["mean_width"]
    return {"n": int(len(y)), "coverage": m["coverage"], "refusal": r,
            "accepted_coverage": (m["coverage"] - r) / (1 - r) if r < 1 else float("nan"),
            "finite_and_correct": m["coverage"] - r,
            "mean_width_pp": 100.0 * w if np.isfinite(w) else float("nan")}


def cycle_order(cycle_ids):
    """Chronological index (0 = first cycle) of each window's cycle; ids look like 'B0005:17'."""
    ids = np.array([str(x) for x in cycle_ids])
    order = sorted(set(ids.tolist()), key=lambda k: int(k.split(":")[1]))
    pos = {k: i for i, k in enumerate(order)}
    return np.array([pos[k] for k in ids]), len(order)


def reference_cycles(n_cycles, m):
    """A labelled reference cycle every m cycles, starting with the first cycle."""
    return list(range(0, n_cycles, m))


def rolling_refs(c, refs, k):
    """The last k reference cycles recorded strictly before cycle c."""
    return [r for r in refs if r < c][-k:] if k > 0 else []


def cycle_median(values, cycle_ids):
    """Per-cycle median of window values -> (sorted unique cycle ids, medians)."""
    ids = np.array([str(x) for x in cycle_ids])
    v = np.asarray(values, dtype=float)
    u = sorted(set(ids.tolist()))
    return u, np.array([np.median(v[ids == k]) for k in u])


def label_gate(lower_soh, floor):
    """Flag windows whose SOH interval reaches below the lowest calibration SOH label."""
    return np.asarray(lower_soh) < floor

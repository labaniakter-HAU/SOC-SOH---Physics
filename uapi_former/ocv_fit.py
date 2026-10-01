"""Reproducible, fold-safe construction of the NMC OCV-SOC lookup table.

The original 13-knot NMC table (dataset._NMC_OCV_PTS) was hand-selected while
looking at all four LOCO cells (B0005/06/07/18), which makes LOCO evaluation
transductive in its feature configuration (reviewer comment R2.2). This module
replaces the manual step with an automated rule that can be restricted to the
training partition of the source cells of each LOCO fold:

  1. For every kept discharge cycle, form the IR-corrected voltage
     V_ocv = V + R0*|I| on samples under load (|I| > min_current).
  2. Label each sample with Coulomb-counted SOC = 1 - Q_out / Q_cycle
     (identical to the SOC label convention in dataset.py).
  3. Interpolate V_ocv at the interior SOC knots (0.05 ... 0.95) per cycle.
  4. Take the median across all kept cycles of all source cells.
  5. Keep the clip endpoints 0.0 -> 2.70 V and 1.0 -> 4.20 V fixed (they are
     the feature clip bounds, not fitted values) and enforce monotonicity.

No SOC/SOH predictions and no held-out-cell data are used.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import numpy as np

SOC_KNOTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                      0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
V_MIN, V_MAX = 2.70, 4.20


def _load_cycles(mat_path: str, cell: str):
    try:
        import mat73
        data = mat73.loadmat(mat_path)
    except Exception:
        from scipy.io import loadmat
        data = loadmat(mat_path, simplify_cells=True)
    cycles = data.get(cell, data.get(cell.lower()))
    if isinstance(cycles, dict) and "cycle" in cycles:
        cycles = cycles["cycle"]
    if not isinstance(cycles, list):
        cycles = [cycles]
    return [c for c in cycles if isinstance(c, dict)
            and str(c.get("type", "")).strip().lower() == "discharge"]


def cycle_ocv_at_knots(cyc: dict, r0: float, min_current: float = 0.5) -> Optional[np.ndarray]:
    d = cyc.get("data", {})
    V = np.asarray(d.get("Voltage_measured", []), float).ravel()
    I = np.asarray(d.get("Current_measured", []), float).ravel()
    t = np.asarray(d.get("Time", []), float).ravel()
    cap = np.asarray(d.get("Capacity", []), float).ravel()
    n = min(len(V), len(I), len(t))
    if n < 10:
        return None
    V, I, t = V[:n], I[:n], t[:n]
    q_cycle = float(cap[-1]) if len(cap) else float(np.sum(np.abs(I[:-1]) * np.diff(t)) / 3600.0)
    if q_cycle < 0.1:
        return None
    dt = np.diff(t, prepend=t[0]); dt[0] = 0.0
    soc = np.clip(1.0 - np.cumsum(np.abs(I) * dt) / 3600.0 / q_cycle, 0.0, 1.0)
    load = np.abs(I) > min_current
    if load.sum() < 10:
        return None
    v_ocv, s = V[load] + r0 * np.abs(I[load]), soc[load]
    order = np.argsort(s)
    s, v_ocv = s[order], v_ocv[order]
    inner = SOC_KNOTS[1:-1]
    ok = (inner >= s.min()) & (inner <= s.max())
    out = np.full(len(inner), np.nan)
    out[ok] = np.interp(inner[ok], s, v_ocv)
    return out


def fit_nmc_ocv_table(data_dir: str, cell_positions: Dict[str, Iterable[int]],
                      r0: float = 0.15) -> np.ndarray:
    """Fit the 13 OCV knots from the given discharge-cycle positions per cell.

    cell_positions maps cell name -> iterable of discharge-cycle positions
    (0-based index into that cell's discharge cycles) allowed for fitting.
    Returns the OCV voltages aligned with SOC_KNOTS.
    """
    rows = []
    for cell, positions in cell_positions.items():
        dcs = _load_cycles(os.path.join(data_dir, f"{cell}.mat"), cell)
        for p in positions:
            if 0 <= p < len(dcs):
                r = cycle_ocv_at_knots(dcs[p], r0)
                if r is not None:
                    rows.append(r)
    if not rows:
        raise RuntimeError("no cycles available for OCV fit")
    med = np.nanmedian(np.vstack(rows), axis=0)
    pts = np.concatenate([[V_MIN], med, [V_MAX]])
    pts = np.clip(np.maximum.accumulate(pts), V_MIN, V_MAX)
    # strictly increasing for np.interp inversion
    for i in range(1, len(pts)):
        if pts[i] <= pts[i - 1]:
            pts[i] = pts[i - 1] + 1e-4
    return pts.astype(np.float64)


def n_discharge_cycles(data_dir: str, cell: str) -> int:
    return len(_load_cycles(os.path.join(data_dir, f"{cell}.mat"), cell))

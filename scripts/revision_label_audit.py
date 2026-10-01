"""R1.4: audit the SOC/SOH label construction exactly as uapi_former/dataset.py builds it.

The loader computes, per discharge cycle,
    SOH = clip(Q_cycle / Q_first, 0.5, 1)          Q_cycle = recorded 'Capacity' field
    SOC = clip(1 - Q_out(t) / (SOH * Q_first), 0, 1),  Q_out = integrated |I| dt
so the SOC denominator equals Q_cycle only when Q_cycle/Q_first lies in [0.5, 1], and SOC
reaches exactly 0 at the end of discharge only when the integrated charge equals the
recorded capacity. This script measures how often each assumption fails, per cell.

Output: results/revision/label_audit.json
"""
import json
import os

import numpy as np

CELLS = ["B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028"]
ROOT = "data/raw/nasa"


def load_cycles(cell):
    path = os.path.join(ROOT, cell + ".mat")
    try:
        import mat73
        data = mat73.loadmat(path)
    except Exception:
        from scipy.io import loadmat
        data = loadmat(path, simplify_cells=True)
    cycles = data.get(cell, data.get(cell.lower()))
    if isinstance(cycles, dict) and "cycle" in cycles:
        cycles = cycles["cycle"]
    if isinstance(cycles, dict):   # mat73 returns a dict of lists
        keys = list(cycles.keys())
        cycles = [{k: cycles[k][i] for k in keys} for i in range(len(cycles[keys[0]]))]
    return cycles if isinstance(cycles, list) else [cycles]


def audit(cell):
    q_first = None
    rows = []
    for cyc in load_cycles(cell):
        if not isinstance(cyc, dict) or str(cyc.get("type", "")).strip().lower() != "discharge":
            continue
        d = cyc.get("data", {})
        I = np.asarray(d.get("Current_measured", []), dtype=np.float64).ravel()
        t = np.asarray(d.get("Time", []), dtype=np.float64).ravel()
        V = np.asarray(d.get("Voltage_measured", []), dtype=np.float64).ravel()
        T = np.asarray(d.get("Temperature_measured", []), dtype=np.float64).ravel()
        cap = np.asarray(d.get("Capacity", []), dtype=np.float64).ravel()
        n = min(len(V), len(I), len(T), len(t))
        if n < 10:
            continue
        I, t = I[:n], t[:n]
        q_cycle = float(cap[-1]) if len(cap) else float(np.sum(np.abs(I[:-1]) * np.diff(t)) / 3600.0)
        if q_cycle < 0.1:
            continue
        if q_first is None:
            q_first = q_cycle
        ratio = q_cycle / q_first
        soh = float(np.clip(ratio, 0.5, 1.0))
        dt = np.diff(t, prepend=t[0]); dt[0] = 0.0
        q_out = np.cumsum(np.abs(I) * dt) / 3600.0
        denom = soh * q_first
        soc_unclipped = 1.0 - q_out / denom
        # time-weighted share of the discharge record where the label sits on a clip bound
        dur = t[-1] - t[0]
        at_zero = float(np.sum(dt[soc_unclipped <= 0.0]) / dur) if dur > 0 else 0.0
        rows.append({
            "ratio": ratio, "soh_clipped_high": ratio > 1.0, "soh_clipped_low": ratio < 0.5,
            "q_out_end_over_q_cycle": float(q_out[-1] / q_cycle),
            "soc_end_unclipped": float(soc_unclipped[-1]),
            "time_share_soc_clipped_at_0": at_zero,
        })
    r = np.array([x["ratio"] for x in rows])
    qe = np.array([x["q_out_end_over_q_cycle"] for x in rows])
    se = np.array([x["soc_end_unclipped"] for x in rows])
    z = np.array([x["time_share_soc_clipped_at_0"] for x in rows])
    return {
        "n_cycles": len(rows),
        "q_first_ah": q_first,
        "n_soh_clipped_high": int(np.sum(r > 1.0)),
        "n_soh_clipped_low": int(np.sum(r < 0.5)),
        "max_ratio": float(r.max()),
        "min_ratio": float(r.min()),
        "q_out_end_over_q_cycle_min_med_max": [float(qe.min()), float(np.median(qe)), float(qe.max())],
        "soc_end_unclipped_min_med_max": [float(se.min()), float(np.median(se)), float(se.max())],
        "n_cycles_soc_hits_0_early": int(np.sum(se < 0.0)),
        "time_share_soc_clipped_at_0_max": float(z.max()),
        "time_share_soc_clipped_at_0_mean": float(z.mean()),
    }


if __name__ == "__main__":
    out = {c: audit(c) for c in CELLS}
    for c, a in out.items():
        print(c, {k: (round(v, 4) if isinstance(v, float) else ([round(x, 4) for x in v] if isinstance(v, list) else v))
                  for k, v in a.items()})
    os.makedirs("results/revision", exist_ok=True)
    json.dump(out, open("results/revision/label_audit.json", "w"), indent=1)
    print("wrote results/revision/label_audit.json")

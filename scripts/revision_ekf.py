"""Revision R2.5a: EKF intervals at matched nominal levels + clean conformal EKF.

Part A (published tuning, B0005+B0007): Gaussian covariance intervals at
  z = 1.645 (90 %), 1.960 (95 %) and 3.0 (99.7 %), SOC per time step and SOH per
  cycle, with SOH reported both vs rated 2 Ah (EKF-internal definition) and vs
  first-cycle capacity (the Transformer's definition).
Part B (clean conformal EKF): noise covariances tuned on B0005 only, split
  conformal quantile of |e|/sigma computed on B0007 only (never used for
  tuning), evaluated on the untouched cells B0006, B0018, B0025-B0028.
Writes results/revision/ekf_revision.json (never overwrites results/ekf_baseline.json).
"""
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former.baselines_ekf import DualEKF, load_discharge_cycles

D = "data/raw/nasa"
Z = {"cov90": 1.645, "cov95": 1.960, "cov3sigma": 3.0}
EVAL = ["B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028"]


def stream(cell, q_soc, r_meas):
    cycles = load_discharge_cycles(os.path.join(D, f"{cell}.mat"), cell)
    cap0 = next((c["capacity"] for c in cycles if np.isfinite(c["capacity"])), 2.0)
    soc_e, soc_sd, soh_e_rated, soh_e_first, soh_sd = [], [], [], [], []
    for c in cycles:
        if not np.isfinite(c["capacity"]) or len(c["V"]) < 20:
            continue
        dt = np.diff(c["t"], prepend=c["t"][0])
        i_a = np.abs(c["I"])
        soc_true = 1.0 - np.cumsum(i_a * dt) / 3600.0 / c["capacity"]
        ekf = DualEKF(capacity_ah=cap0, soc0=1.0, q_soc=q_soc, r_meas=r_meas)
        for k in range(len(c["V"])):
            s, sd = ekf.step(c["V"][k], i_a[k], max(dt[k], 1e-3))
            soc_e.append(s - soc_true[k]); soc_sd.append(sd)
        soh_e_rated.append(ekf.soh - c["capacity"] / 2.0)
        soh_e_first.append(ekf.soh * 2.0 / cap0 - c["capacity"] / cap0)
        soh_sd.append(ekf.soh_std)
    return {k: np.asarray(v) for k, v in dict(soc_e=soc_e, soc_sd=soc_sd, soh_e_rated=soh_e_rated,
                                              soh_e_first=soh_e_first, soh_sd=soh_sd).items()}


def summarize(r, soc_q=None):
    out = {"soc_rmse": float(100 * np.sqrt(np.mean(r["soc_e"] ** 2))),
           "soh_rmse_vs_rated": float(100 * np.sqrt(np.mean(r["soh_e_rated"] ** 2))),
           "soh_rmse_vs_first_cycle": float(100 * np.sqrt(np.mean(r["soh_e_first"] ** 2))),
           "n_steps": int(len(r["soc_e"])), "n_cycles": int(len(r["soh_e_rated"]))}
    for name, z in Z.items():
        out[f"soc_{name}"] = float(np.mean(np.abs(r["soc_e"]) <= z * r["soc_sd"]))
        out[f"soh_{name}"] = float(np.mean(np.abs(r["soh_e_rated"]) <= z * r["soh_sd"]))
    if soc_q is not None:
        out["soc_conformal90"] = float(np.mean(np.abs(r["soc_e"]) <= soc_q * r["soc_sd"]))
        out["soc_conformal90_mean_width_pp"] = float(100 * np.mean(2 * soc_q * r["soc_sd"]))
    return out


def tune(cells):
    best, best_rmse = None, 1e9
    for q, rm in itertools.product([1e-8, 1e-7, 1e-6], [1e-4, 1e-3, 1e-2]):
        rmse = np.mean([summarize(stream(c, q, rm))["soc_rmse"] for c in cells])
        if rmse < best_rmse:
            best, best_rmse = (q, rm), rmse
    return best


def main():
    res = {}
    q, rm = tune(["B0005", "B0007"])
    res["A_published_tuning"] = {"tuned_on": ["B0005", "B0007"], "q_soc": q, "r_meas": rm,
                                 "cells": {c: summarize(stream(c, q, rm)) for c in EVAL}}
    q2, rm2 = tune(["B0005"])
    cal = stream("B0007", q2, rm2)
    scores = np.abs(cal["soc_e"]) / np.maximum(cal["soc_sd"], 1e-12)
    n = len(scores)
    soc_q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.9) / n)))
    res["B_clean_conformal"] = {"tuned_on": ["B0005"], "calibrated_on": ["B0007"], "q_soc": q2, "r_meas": rm2,
                                "soc_conformal_q90": soc_q, "n_cal_steps": int(n),
                                "cells": {c: summarize(stream(c, q2, rm2), soc_q)
                                          for c in ["B0006", "B0018", "B0025", "B0026", "B0027", "B0028"]}}
    os.makedirs("results/revision", exist_ok=True)
    json.dump(res, open("results/revision/ekf_revision.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

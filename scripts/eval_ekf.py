"""EKF control-baseline evaluation over the shift hierarchy.

For each NASA cell (B0005/06/07/18 = LOCO-equivalent; B0025-28 = cross
protocol), streams discharge cycles through DualEKF and reports SOC/SOH RMSE
plus empirical coverage of the native +/-3-sigma covariance intervals.
Noise covariances (q_soc, r_meas) are grid-tuned ONCE on B0005+B0007 training
cells and then frozen for every evaluation (no per-cell tuning -- control
condition, not a competitor). Writes results/ekf_baseline.json.
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from uapi_former.baselines_ekf import DualEKF, load_discharge_cycles

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
PROTO_CELLS = ["B0025", "B0026", "B0027", "B0028"]


def run_cell(mat_path, cell, q_soc, r_meas):
    cycles = load_discharge_cycles(mat_path, cell)
    cap0 = next((c["capacity"] for c in cycles if np.isfinite(c["capacity"])), 2.0)
    errs_soc, errs_soh, cov_soc, cov_soh = [], [], [], []
    for c in cycles:
        if not np.isfinite(c["capacity"]) or len(c["V"]) < 20:
            continue
        dt = np.diff(c["t"], prepend=c["t"][0])
        i_a = np.abs(c["I"])                      # discharge magnitude
        q_ah = np.cumsum(i_a * dt) / 3600.0
        soc_true = 1.0 - q_ah / c["capacity"]
        soh_true = c["capacity"] / 2.0            # C_NOMINAL_AH
        ekf = DualEKF(capacity_ah=cap0, soc0=1.0, q_soc=q_soc, r_meas=r_meas)
        for k in range(len(c["V"])):
            s, s_std = ekf.step(c["V"][k], i_a[k], max(dt[k], 1e-3))
            errs_soc.append(s - soc_true[k])
            cov_soc.append(abs(s - soc_true[k]) <= 3 * s_std)
        errs_soh.append(ekf.soh - soh_true)
        cov_soh.append(abs(ekf.soh - soh_true) <= 3 * ekf.soh_std)
    return {"soc_rmse": float(np.sqrt(np.mean(np.square(errs_soc)))) * 100,
            "soh_rmse": float(np.sqrt(np.mean(np.square(errs_soh)))) * 100,
            "soc_cov3sigma": float(np.mean(cov_soc)),
            "soh_cov3sigma": float(np.mean(cov_soh)),
            "n_steps": len(errs_soc), "n_cycles": len(errs_soh)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw/nasa")
    p.add_argument("--out", default="results/ekf_baseline.json")
    args = p.parse_args()
    # coarse grid tune on B0005+B0007 (train-cell discipline)
    best, best_rmse = None, 1e9
    for q_soc, r_meas in itertools.product([1e-8, 1e-7, 1e-6], [1e-4, 1e-3, 1e-2]):
        r = [run_cell(os.path.join(args.data_dir, f"{c}.mat"), c, q_soc, r_meas)
             for c in ("B0005", "B0007")]
        rmse = np.mean([x["soc_rmse"] for x in r])
        if rmse < best_rmse:
            best, best_rmse = (q_soc, r_meas), rmse
    q_soc, r_meas = best
    out = {"tuned_on": ["B0005", "B0007"], "q_soc": q_soc, "r_meas": r_meas}
    for cell in NASA_CELLS + PROTO_CELLS:
        path = os.path.join(args.data_dir, f"{cell}.mat")
        if os.path.exists(path):
            out[cell] = run_cell(path, cell, q_soc, r_meas)
            print(cell, out[cell])
    assert not os.path.exists(args.out), f"refusing to overwrite {args.out}"
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

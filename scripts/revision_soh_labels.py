"""Revision R1.4: consequences of assigning a cycle-level SOH label to windows.

Canonical random-split checkpoint (checkpoints/nasa_v12_nig/best_calibrated.pt),
random-split TEST partition only (never used in training or model selection).

Reports
  1. how many windows share each SOH label (windows per cycle),
  2. SOH absolute error vs relative window position within the discharge,
  3. SOH S0 coverage at the window level vs the cycle level (one decision per
     cycle: the cycle is covered if the interval of its LAST window covers,
     and alternatively if the median-width window covers), with the
     calibration quantile taken from the calibration half of the test cycles
     (clean calibration, see R2.8) so that no selection data are reused.
Writes results/revision/soh_label_analysis.json
"""
import json, os, sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former.dataset import NASABatteryDataset
from uapi_former import shift_conformal as sc
from scripts.shift_conformal_study import load_model, collect


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nig-seed", type=int, default=None,
                    help="SP4: checkpoints/nasa_v12_nig_seed{k}; writes results/revision/seeds/soh_label_analysis_s{k}.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ck = "checkpoints/nasa_v12_nig/best_calibrated.pt" if a.nig_seed is None else f"checkpoints/nasa_v12_nig_seed{a.nig_seed}/best.pt"
    out_path = ("results/revision/soh_label_analysis.json" if a.nig_seed is None
                else f"results/revision/seeds/soh_label_analysis_s{a.nig_seed}.json")
    model, meta = load_model(ck, dev)
    ds = NASABatteryDataset("data/raw/nasa", split="test", split_mode="intra_cell_random")
    out = collect(model, DataLoader(ds, batch_size=256, shuffle=False), dev, meta)
    cyc = np.array([f"{c}:{p}" for c, p, _ in ds.meta])
    start = np.array([st for _, _, st in ds.meta])
    soh, pred = out["soh"].numpy(), out["hg"].numpy()
    err = np.abs(pred - soh)

    # windows per cycle and relative position
    groups = defaultdict(list)
    for i, c in enumerate(cyc):
        groups[c].append(i)
    n_per = np.array([len(v) for v in groups.values()])
    relpos = np.empty(len(cyc))
    for idx in groups.values():
        st = start[idx]; relpos[idx] = (st - st.min()) / max(1, st.max() - st.min())

    bins = np.linspace(0, 1, 6)
    pos_tab = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (relpos >= lo) & (relpos <= hi if hi == 1 else relpos < hi)
        pos_tab.append({"relpos": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                        "soh_mae_pct": float(100 * err[m].mean()),
                        "soh_rmse_pct": float(100 * np.sqrt((err[m] ** 2).mean()))})

    # clean calibration: split test CYCLES in half (seeded)
    ukeys = sorted(groups)
    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(ukeys))
    cal_keys = set(ukeys[k] for k in perm[: len(ukeys) // 2])
    is_cal = np.array([c in cal_keys for c in cyc])
    s_all, sigma = sc.nonconformity(out["hg"], out["hv"], out["ha"], out["hb"], out["soh"])
    s_all, sigma = s_all.numpy(), sigma.numpy()
    res = {"n_windows_test": int(len(cyc)), "n_cycles_test": int(len(groups)),
           "windows_per_cycle": {"mean": float(n_per.mean()), "min": int(n_per.min()), "max": int(n_per.max())},
           "soh_error_vs_position": pos_tab, "coverage": {}}
    for cov in (0.90, 0.95):
        q = sc.vanilla_q(torch.tensor(s_all[is_cal]), cov)
        ev = ~is_cal
        covered = np.abs(pred - soh) <= q * sigma
        win_cov = float(covered[ev].mean())
        last_cov, per_cycle_frac = [], []
        for c in set(cyc[ev]):
            idx = np.array(groups[c])
            last = idx[np.argmax(start[idx])]
            last_cov.append(covered[last]); per_cycle_frac.append(covered[idx].mean())
        per_cycle_frac = np.array(per_cycle_frac)
        res["coverage"][f"cov{int(cov*100)}"] = {
            "q": q, "n_cal_windows": int(is_cal.sum()), "n_eval_windows": int(ev.sum()),
            "n_eval_cycles": int(len(per_cycle_frac)),
            "window_level": win_cov,
            "cycle_level_last_window": float(np.mean(last_cov)),
            "cycles_fully_covered": float(np.mean(per_cycle_frac == 1.0)),
            "cycles_fully_missed": float(np.mean(per_cycle_frac == 0.0)),
            "per_cycle_coverage_p10_p50_p90": [float(x) for x in np.percentile(per_cycle_frac, [10, 50, 90])],
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

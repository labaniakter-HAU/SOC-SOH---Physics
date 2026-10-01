"""Revision R2.3: sequential (prefix) density-ratio estimation instead of the
transductive full-target-batch construction.

For each clean LOCO fold, target windows are ordered by cycle. The S1
discriminator is fitted on the calibration latents versus the target latents of
the FIRST k cycles only, and intervals are then issued for windows from strictly
LATER cycles (k = 1, 5, 20). The transductive variant (discriminator fitted on
the whole target batch, including the evaluated windows) is re-evaluated on the
same later-cycle windows so the comparison is like-for-like, and S0 is included
as the no-correction reference.

Reads the caches written by scripts/revision_conformal_clean.py
(results/revision/cache/loco_clean_s{seed}_{cell}.pt).
Writes results/revision/sequential_s1.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from uapi_former.conformal_extras import restrict_active

CELLS = ["B0005", "B0006", "B0007", "B0018"]
KS = [1, 5, 20]
COV = 0.90


def metrics(y, g, sigma, q):
    m = sc.interval_metrics(y, g, sigma, q)
    r = m.get("frac_uncertifiable", 0.0) or 0.0
    m["refusal"] = r
    m["accepted_coverage"] = (m["coverage"] - r) / (1 - r) if r < 1 else float("nan")
    m["finite_and_correct"] = m["coverage"] - r
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--active", action="store_true",
                   help="calibrate and evaluate on windows with SOC label > 0 only (spec 2026-09-24 §4.1)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = {}
    for cell in CELLS:
        c = torch.load(f"results/revision/cache/loco_clean_s{a.seed}_{cell}.pt",
                       map_location="cpu", weights_only=False)
        cal, tgt = c["cal"], c["tgt"]
        cyc = np.array([str(x) for x in c["tgt_cycle"]])
        if a.active:
            cal, tgt, cyc = restrict_active(cal, tgt, cyc)
        order = sorted(set(cyc), key=lambda k: int(k.split(":")[1]))
        pos = {k: i for i, k in enumerate(order)}
        idx = np.array([pos[k] for k in cyc])
        s_cal, _ = sc.nonconformity(cal["sg"], cal["sv"], cal["sa"], cal["sb"], cal["soc"])
        entry = {"n_target_cycles": len(order)}
        for k in KS:
            fit_mask = idx < k                    # first k cycles: unlabelled fitting data
            ev_mask = idx >= max(KS)              # identical evaluation set for every k
            if fit_mask.sum() < 20 or ev_mask.sum() < 50:
                continue
            z_fit = tgt["rep"][torch.as_tensor(fit_mask)]
            ev = {kk: v[torch.as_tensor(ev_mask)] for kk, v in tgt.items()}
            _, sigma = sc.nonconformity(ev["sg"], ev["sv"], ev["sa"], ev["sb"], ev["soc"])
            w_cal = sc.logistic_density_ratio(cal["rep"], z_fit)
            w_ev = sc.logistic_density_ratio(cal["rep"], z_fit, query=ev["rep"])
            q = sc.weighted_q(s_cal, w_cal, COV, w_test=w_ev)
            entry[f"seq_k{k}"] = metrics(ev["soc"], ev["sg"], sigma, q)
            entry[f"seq_k{k}"]["n_fit_windows"] = int(fit_mask.sum())
            entry[f"seq_k{k}"]["ess"] = sc.effective_sample_size(w_cal)
        # references on the same evaluation window set
        ev_mask = idx >= max(KS)
        ev = {kk: v[torch.as_tensor(ev_mask)] for kk, v in tgt.items()}
        _, sigma = sc.nonconformity(ev["sg"], ev["sv"], ev["sa"], ev["sb"], ev["soc"])
        entry["S0"] = metrics(ev["soc"], ev["sg"], sigma, sc.vanilla_q(s_cal, COV))
        w_cal_t = sc.logistic_density_ratio(cal["rep"], tgt["rep"])
        w_ev_t = sc.logistic_density_ratio(cal["rep"], tgt["rep"], query=ev["rep"])
        entry["transductive"] = metrics(ev["soc"], ev["sg"], sigma,
                                        sc.weighted_q(s_cal, w_cal_t, COV, w_test=w_ev_t))
        entry["n_eval_windows"] = int(ev_mask.sum())
        # Why sequential fitting fails: are the first k cycles representative of the
        # evaluated windows? SOH labels are used here only as a diagnostic (the method
        # never sees them); the AUCs are the in-sample discriminator AUC used elsewhere.
        soh = tgt["soh"].numpy()
        z_ev = tgt["rep"][torch.as_tensor(ev_mask)]
        entry["mechanism"] = {
            "cal_vs_eval_auc": sc.discriminator_auc(cal["rep"], z_ev),
            **{f"k{k}": {"fit_soh_min": float(soh[idx < k].min()),
                         "eval_below_fit_soh_pct": float(100 * (soh[ev_mask] < soh[idx < k].min()).mean()),
                         "fit_vs_eval_auc": sc.discriminator_auc(tgt["rep"][torch.as_tensor(idx < k)], z_ev)}
               for k in KS}}
        out[cell] = entry
        print(cell, {k: round(100 * v["accepted_coverage"], 1) for k, v in entry.items()
                     if isinstance(v, dict) and "accepted_coverage" in v}, flush=True)
    path = a.out or f"results/revision/sequential_s1_s{a.seed}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

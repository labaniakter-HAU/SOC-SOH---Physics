"""Revision R2.4: mechanism test for the B0006 LOCO failure.

Hypothesis: B0006 fails because its labels leave the range the source cells
span (label/conditional shift), not only because its inputs move (covariate shift).
SOH labels are cycle capacity / first-cycle capacity. B0006 falls to 0.567 of its
first-cycle capacity; the training cycles of its source cells never go below 0.693
(B0005's minimum), so 33.8 % of B0006's windows lie below the source SOH range.

For every clean LOCO fold we split the target windows by whether the cycle's SOH
lies inside or below the SOH range spanned by that fold's TRAINING cycles, and
report S0/S1 coverage, error and latent-distance diagnostics in each stratum, plus
the Spearman correlation of four per-window signals (kNN latent distance to the
calibration set, density ratio, evidential sigma, epistemic variance) with the
absolute SOC error. SOH labels are used only as a diagnostic.

Writes results/revision/b0006_diagnosis_s{seed}.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from uapi_former.dataset import NASABatteryDataset

CELLS = ["B0005", "B0006", "B0007", "B0018"]
COV = 0.90


def source_soh_range(cell, seed):
    """SOH range covered by the TRAINING cycles of the fold's source cells."""
    # the fold's training set holds the training cycles of all three source cells;
    # q_norm only changes the SOC denominator, not the SOH label
    ds = NASABatteryDataset("data/raw/nasa", split="train", split_mode="loco", held_out_cell=cell,
                            loco_split="random3", loco_seed=seed, q_norm="rated")
    sohs = [float(x[3]) for x in ds.samples]
    return min(sohs), max(sohs)


def strat(y, g, sigma, q, mask):
    m = sc.interval_metrics(y[mask], g[mask], sigma[mask], q if not isinstance(q, torch.Tensor) else q[mask])
    m["n"] = int(mask.sum())
    m["rmse_pct"] = float(100 * torch.sqrt(((y[mask] - g[mask]) ** 2).mean()))
    return m


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = {}
    for cell in CELLS:
        c = torch.load(f"results/revision/cache/loco_clean_s{a.seed}_{cell}.pt",
                       map_location="cpu", weights_only=False)
        cal, tgt = c["cal"], c["tgt"]
        lo, hi = source_soh_range(cell, a.seed)
        s_cal, _ = sc.nonconformity(cal["sg"], cal["sv"], cal["sa"], cal["sb"], cal["soc"])
        q0 = sc.vanilla_q(s_cal, COV)
        w_cal = sc.logistic_density_ratio(cal["rep"], tgt["rep"])
        w_tgt = sc.logistic_density_ratio(cal["rep"], tgt["rep"], query=tgt["rep"])
        q1 = sc.weighted_q(s_cal, w_cal, COV, w_test=w_tgt)
        _, sigma = sc.nonconformity(tgt["sg"], tgt["sv"], tgt["sa"], tgt["sb"], tgt["soc"])
        inside = (tgt["soh"] >= lo) & (tgt["soh"] <= hi)
        below = tgt["soh"] < lo
        # latent-distance diagnostic: mean distance to 10 nearest calibration latents
        d = torch.cdist(tgt["rep"].float(), cal["rep"].float())
        knn = d.topk(10, largest=False).values.mean(1)
        e = {"source_soh_range": [lo, hi],
             "frac_target_below_source_range": float(below.float().mean()),
             "target_soh_min": float(tgt["soh"].min())}
        # which per-window signals track the error (Spearman rank correlation with the
        # absolute SOC error over all target windows of the fold)
        abs_err = torch.abs(tgt["soc"] - tgt["sg"]).numpy()
        e["spearman_abs_err"] = {
            nm: float(spearmanr(v, abs_err)[0]) for nm, v in (
                ("knn_dist", knn.numpy()), ("density_ratio", w_tgt.numpy()), ("sigma", sigma.numpy()),
                ("epistemic", sc.epistemic_variance(tgt["sv"], tgt["sa"], tgt["sb"]).numpy()))}
        for nm, m in (("inside", inside), ("below", below)):
            if m.sum() < 20:
                e[nm] = None
                continue
            e[nm] = {"S0": strat(tgt["soc"], tgt["sg"], sigma, q0, m),
                     "S1": strat(tgt["soc"], tgt["sg"], sigma, q1, m),
                     "mean_knn_dist": float(knn[m].mean()),
                     "mean_density_ratio": float(w_tgt[m].mean()),
                     "mean_epistemic": float(sc.epistemic_variance(tgt["sv"], tgt["sa"], tgt["sb"])[m].mean())}
        out[cell] = e
        print(cell, f"below={100*e['frac_target_below_source_range']:.1f}%",
              {k: (round(100 * e[k]["S0"]["coverage"], 1), round(e[k]["mean_knn_dist"], 2))
               for k in ("inside", "below") if e.get(k)}, flush=True)
    os.makedirs("results/revision", exist_ok=True)
    path = f"results/revision/b0006_diagnosis_s{a.seed}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

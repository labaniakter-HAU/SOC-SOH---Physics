"""R1.4 follow-up: how much of each evaluation set carries a SOC label clipped at 0, and does
that stratum drive the reported SOC error and S0 coverage?

The loader keeps each discharge record to its end, including the rest period after the
cut-off voltage. Integrated charge exceeds the recorded capacity by 0.2-3.1 %, so the
Coulomb-counted SOC reaches 0 just before cut-off and every later window carries SOC = 0
(scripts/revision_label_audit.py). This script splits every cached evaluation set at that
boundary and recomputes SOC RMSE and S0 coverage (90 % nominal) in each stratum, using the
same score and quantile functions as the paper (uapi_former/shift_conformal.py).

Reads results/revision/cache/{random,protocol_B00xx,loco_clean_s{seed}_B00xx}.pt
Writes results/revision/rest_windows.json
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, ".")
from uapi_former.shift_conformal import nonconformity, vanilla_q  # noqa: E402

CACHE = "results/revision/cache/{}.pt"


def s0(d):
    c, t = d["cal"], d["tgt"]
    s_cal, _ = nonconformity(c["sg"], c["sv"], c["sa"], c["sb"], c["soc"])
    q = vanilla_q(s_cal, 0.90)
    s_t, _ = nonconformity(t["sg"], t["sv"], t["sa"], t["sb"], t["soc"])
    cov = (s_t <= q).float()
    err = (t["sg"] - t["soc"]) * 100.0
    zero = t["soc"] <= 1e-6

    def block(m):
        n = int(m.sum())
        if n == 0:
            return {"n": 0}
        y, g = t["soc"][m].double(), t["sg"][m].double()
        r = float(torch.corrcoef(torch.stack([y, g]))[0, 1]) if n > 2 and y.std() > 0 else float("nan")
        return {"n": n, "soc_rmse_pct": float(err[m].pow(2).mean().sqrt()), "soc_r": r,
                "s0_cov90_pct": float(cov[m].mean() * 100), "mean_pred_soc_pct": float(t["sg"][m].mean() * 100)}

    return {"share_label_zero_pct": float(zero.float().mean() * 100),
            "all": block(torch.ones_like(zero)), "label_zero": block(zero), "label_positive": block(~zero)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--nig-seeds", default="", help="SP4: random_s{k}/protocol_s{k}_* caches of the v12-NIG seeds")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    nig = [int(x) for x in a.nig_seeds.split(",") if x != ""]
    out = {"random": s0(torch.load(CACHE.format("random"), weights_only=False))}
    for c in ["B0025", "B0026", "B0027", "B0028"]:
        out[f"protocol_{c}"] = s0(torch.load(CACHE.format(f"protocol_{c}"), weights_only=False))
    for seed in seeds:
        for c in ["B0005", "B0006", "B0007", "B0018"]:
            out[f"loco_s{seed}_{c}"] = s0(torch.load(CACHE.format(f"loco_clean_s{seed}_{c}"), weights_only=False))
    for k in nig:
        out[f"random_s{k}"] = s0(torch.load(CACHE.format(f"random_s{k}"), weights_only=False))
        for c in ["B0025", "B0026", "B0027", "B0028"]:
            out[f"protocol_s{k}_{c}"] = s0(torch.load(CACHE.format(f"protocol_s{k}_{c}"), weights_only=False))
    for k, v in out.items():
        a, z, p = v["all"], v["label_zero"], v["label_positive"]
        print(f"{k:18s} zero-share {v['share_label_zero_pct']:5.1f}% | all RMSE {a['soc_rmse_pct']:6.2f} cov {a['s0_cov90_pct']:5.1f}"
              f" | SOC=0: RMSE {z.get('soc_rmse_pct', float('nan')):6.2f} cov {z.get('s0_cov90_pct', float('nan')):5.1f}"
              f" pred {z.get('mean_pred_soc_pct', float('nan')):5.1f}"
              f" | SOC>0: RMSE {p['soc_rmse_pct']:6.2f} cov {p['s0_cov90_pct']:5.1f}")
    json.dump(out, open("results/revision/rest_windows.json", "w"), indent=1)
    print("wrote results/revision/rest_windows.json")


if __name__ == "__main__":
    main()

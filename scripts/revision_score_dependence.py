"""R1.6 follow-up: how much of the coverage collapse depends on the conformal score?

Table 5 uses the sigma-normalized NIG score |y - g| / sigma. The baselines-under-shift
analysis (Table 7 and the cross-protocol baseline table) uses the plain absolute residual
|y - g|, because the baselines emit no scale. This script evaluates BOTH scores on the SAME
canonical v12-NIG predictions and the same calibration partitions (the cached entries behind
Table 5), so the effect of the score is isolated from the effect of the checkpoint.

Reads results/revision/cache/{random,protocol_B00xx,loco_clean_s{seed}_B00xx}.pt
Writes results/revision/score_dependence.json
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, ".")
from uapi_former.shift_conformal import nonconformity, vanilla_q  # noqa: E402

CACHE = "results/revision/cache/{}.pt"


def both(name):
    d = torch.load(CACHE.format(name), weights_only=False)
    c, t = d["cal"], d["tgt"]
    s_cal, sig_cal = nonconformity(c["sg"], c["sv"], c["sa"], c["sb"], c["soc"])
    s_tgt, sig_tgt = nonconformity(t["sg"], t["sv"], t["sa"], t["sb"], t["soc"])
    q_sig = vanilla_q(s_cal, 0.90)
    q_abs = vanilla_q((c["soc"] - c["sg"]).abs(), 0.90)
    r_tgt = (t["soc"] - t["sg"]).abs()
    pos = t["soc"] > 1e-6                     # SOC > 0: excludes the end-of-discharge/rest windows (R1.4)
    cov = lambda hit, m: float(hit[m].float().mean() * 100) if m.any() else float("nan")
    return {"sigma_normalized": {"s0_cov90_pct": float((s_tgt <= q_sig).float().mean() * 100),
                                 "s0_cov90_pct_soc_pos": cov(s_tgt <= q_sig, pos),
                                 "mean_width_pp": float((2 * q_sig * sig_tgt).mean() * 100)},
            "absolute_residual": {"s0_cov90_pct": float((r_tgt <= q_abs).float().mean() * 100),
                                  "s0_cov90_pct_soc_pos": cov(r_tgt <= q_abs, pos),
                                  "width_pp": float(2 * q_abs * 100)},
            "sigma_ratio_tgt_over_cal_median": float(sig_tgt.median() / sig_cal.median()),
            "sigma_ratio_tgt_over_cal_median_soc_pos": float(sig_tgt[pos].median() / sig_cal.median()) if pos.any() else float("nan")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--nig-seeds", default="", help="SP4: random_s{k}/protocol_s{k}_* caches of the v12-NIG seeds")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    nig = [int(x) for x in a.nig_seeds.split(",") if x != ""]
    out = {"random": both("random")}
    for c in ("B0025", "B0026", "B0027", "B0028"):
        out[f"protocol_{c}"] = both(f"protocol_{c}")
    for s in seeds:
        for c in ("B0005", "B0006", "B0007", "B0018"):
            out[f"loco_s{s}_{c}"] = both(f"loco_clean_s{s}_{c}")
    for k in nig:
        out[f"random_s{k}"] = both(f"random_s{k}")
        for c in ("B0025", "B0026", "B0027", "B0028"):
            out[f"protocol_s{k}_{c}"] = both(f"protocol_s{k}_{c}")
    for k, v in out.items():
        print(f"{k:16s} sigma-norm {v['sigma_normalized']['s0_cov90_pct']:5.1f}% (w {v['sigma_normalized']['mean_width_pp']:5.2f})"
              f" | abs-resid {v['absolute_residual']['s0_cov90_pct']:5.1f}% (w {v['absolute_residual']['width_pp']:5.2f})"
              f" | sigma tgt/cal {v['sigma_ratio_tgt_over_cal_median']:.2f}"
              f" || SOC>0: sigma-norm {v['sigma_normalized']['s0_cov90_pct_soc_pos']:5.1f}% abs {v['absolute_residual']['s0_cov90_pct_soc_pos']:5.1f}%"
              f" sigma ratio {v['sigma_ratio_tgt_over_cal_median_soc_pos']:.2f}")
    json.dump(out, open("results/revision/score_dependence.json", "w"), indent=1)
    print("wrote results/revision/score_dependence.json")


if __name__ == "__main__":
    main()

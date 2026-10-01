"""Revision R2.4 (part 2): can the B0006 failure be DETECTED from source data only?

The diagnosis shows B0006's under-coverage concentrates in cycles whose SOH lies
below anything the source cells reach. This script tests a support gate whose
threshold is derived exclusively from the source calibration partition:

  tau = quantile_p( mean distance of each calibration latent to its 10 nearest
                    OTHER calibration latents )

A target window is refused when its mean distance to the 10 nearest calibration
latents exceeds tau. No target labels and no target statistics enter the rule.
Reported per fold: refusal rate, accepted-point coverage, the fraction of the
out-of-source-SOH-range stratum that is refused (detection rate), and the
fraction of in-range windows refused (false-alarm rate).

Writes results/revision/support_gate_s{seed}.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from scripts.revision_b0006_diagnosis import source_soh_range, CELLS

COV = 0.90
PCTS = [95.0, 99.0, 99.9]


def knn_mean(a, b, k=10, exclude_self=False):
    d = torch.cdist(a.float(), b.float())
    kk = k + 1 if exclude_self else k
    v = d.topk(kk, largest=False).values
    if exclude_self:
        v = v[:, 1:]
    return v.mean(1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = {}
    for cell in CELLS:
        c = torch.load(f"results/revision/cache/loco_clean_s{a.seed}_{cell}.pt",
                       map_location="cpu", weights_only=False)
        cal, tgt = c["cal"], c["tgt"]
        lo, _ = source_soh_range(cell, a.seed)
        below = (tgt["soh"] < lo).numpy()
        s_cal, _ = sc.nonconformity(cal["sg"], cal["sv"], cal["sa"], cal["sb"], cal["soc"])
        q0 = sc.vanilla_q(s_cal, COV)
        _, sigma = sc.nonconformity(tgt["sg"], tgt["sv"], tgt["sa"], tgt["sb"], tgt["soc"])
        covered = (torch.abs(tgt["soc"] - tgt["sg"]) <= q0 * sigma).numpy()
        d_cal = knn_mean(cal["rep"], cal["rep"], exclude_self=True).numpy()
        d_tgt = knn_mean(tgt["rep"], cal["rep"]).numpy()
        e = {"frac_below_source_soh": float(below.mean()), "ungated_coverage": float(covered.mean())}
        for pct in PCTS:
            tau = float(np.percentile(d_cal, pct))
            refuse = d_tgt > tau
            acc = ~refuse
            e[f"p{pct}"] = {
                "tau": tau,
                "refusal_rate": float(refuse.mean()),
                "accepted_coverage": float(covered[acc].mean()) if acc.sum() else float("nan"),
                "n_accepted": int(acc.sum()),
                "detection_rate_below": float(refuse[below].mean()) if below.sum() else None,
                "false_alarm_rate_inside": float(refuse[~below].mean()) if (~below).sum() else None,
            }
        out[cell] = e
        print(cell, f"below={100*e['frac_below_source_soh']:.1f}% ungated={100*e['ungated_coverage']:.1f}",
              {f"p{p_}": (round(100 * e[f'p{p_}']['refusal_rate'], 1),
                          round(100 * e[f'p{p_}']['accepted_coverage'], 1)) for p_ in PCTS}, flush=True)
    path = f"results/revision/support_gate_s{a.seed}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

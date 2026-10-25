"""R1.4 follow-up: how much of each evaluation set carries a SOC label clipped at 0, and does
that stratum drive the reported SOC error and S0 coverage?

The loader keeps each discharge record to its end, including the rest period after the
cut-off voltage. Integrated charge exceeds the recorded capacity by 0.2-3.1 %, so the
Coulomb-counted SOC reaches 0 just before cut-off and every later window carries SOC = 0
(scripts/revision_label_audit.py). This script splits every cached evaluation set at that
boundary and recomputes SOC RMSE and S0 coverage (90 % nominal) in each stratum, using the
same score and quantile functions as the paper (uapi_former/shift_conformal.py).

Processes round 1 (R2.6, R2.9): also the stratum SOC > 10 %, window-level SOH S0 coverage, and a
95 % interval for every coverage from a two-sample cycle-block bootstrap: calibration cycles and
evaluation cycles are resampled with replacement (windows of one cycle stay together, since the
cycle is the exchangeable unit), the S0 quantile is recomputed on each calibration resample, and
coverage is recomputed on each evaluation resample. `boot_share_ge_nominal` is the share of the
B resamples with coverage >= 90 % (a one-sided bootstrap p-value against the nominal level).

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
B = 2000  # bootstrap resamples
NOMINAL = 0.90


def cycle_groups(cycles):
    idx = {}
    for i, c in enumerate(cycles):
        idx.setdefault(str(c), []).append(i)
    return [torch.tensor(v) for v in idx.values()]


def bootstrap(s_cal, cal_cycle, s_t, tgt_cycle, masks):
    """95 % interval and share >= nominal of S0 coverage on each mask, resampling whole cycles."""
    gen = torch.Generator().manual_seed(0)
    cg, tg = cycle_groups(cal_cycle), cycle_groups(tgt_cycle)
    draws = {k: [] for k in masks}
    for _ in range(B):
        ci = torch.cat([cg[j] for j in torch.randint(len(cg), (len(cg),), generator=gen)])
        ti = torch.cat([tg[j] for j in torch.randint(len(tg), (len(tg),), generator=gen)])
        hit = (s_t[ti] <= vanilla_q(s_cal[ci], NOMINAL)).float()
        for k, m in masks.items():
            mm = m[ti]
            if mm.any():
                draws[k].append(float(hit[mm].mean()))
    out = {}
    for k, v in draws.items():
        if v:
            v = torch.tensor(v, dtype=torch.float64)
            out[k] = {"ci95_pct": [float(torch.quantile(v, 0.025)) * 100, float(torch.quantile(v, 0.975)) * 100],
                      "boot_share_ge_nominal": float((v >= NOMINAL).double().mean()), "n_boot": len(v)}
    return out


def s0(d, cal_cycle=None):
    c, t = d["cal"], d["tgt"]
    cal_cycle = d.get("cal_cycle", cal_cycle)
    s_cal, _ = nonconformity(c["sg"], c["sv"], c["sa"], c["sb"], c["soc"])
    q = vanilla_q(s_cal, 0.90)
    s_t, _ = nonconformity(t["sg"], t["sv"], t["sa"], t["sb"], t["soc"])
    cov = (s_t <= q).float()
    err = (t["sg"] - t["soc"]) * 100.0
    zero = t["soc"] <= 1e-6
    gt10 = t["soc"] > 0.10

    def block(m):
        n = int(m.sum())
        if n == 0:
            return {"n": 0}
        y, g = t["soc"][m].double(), t["sg"][m].double()
        r = float(torch.corrcoef(torch.stack([y, g]))[0, 1]) if n > 2 and y.std() > 0 else float("nan")
        return {"n": n, "soc_rmse_pct": float(err[m].pow(2).mean().sqrt()), "soc_r": r,
                "s0_cov90_pct": float(cov[m].mean() * 100), "mean_pred_soc_pct": float(t["sg"][m].mean() * 100)}

    h_cal, _ = nonconformity(c["hg"], c["hv"], c["ha"], c["hb"], c["soh"])
    h_t, _ = nonconformity(t["hg"], t["hv"], t["ha"], t["hb"], t["soh"])
    every = torch.ones_like(zero)
    out = {"share_label_zero_pct": float(zero.float().mean() * 100),
           "all": block(every), "label_zero": block(zero), "label_positive": block(~zero), "label_gt10": block(gt10),
           "soh_all": {"n": int(every.sum()), "s0_cov90_pct": float((h_t <= vanilla_q(h_cal, 0.90)).float().mean() * 100)}}
    soc_ci = bootstrap(s_cal, cal_cycle, s_t, d["tgt_cycle"], {"all": every, "label_positive": ~zero, "label_gt10": gt10})
    for k, v in soc_ci.items():
        out[k].update(v)
    out["soh_all"].update(bootstrap(h_cal, cal_cycle, h_t, d["tgt_cycle"], {"all": every})["all"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--nig-seeds", default="", help="SP4: random_s{k}/protocol_s{k}_* caches of the v12-NIG seeds")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    nig = [int(x) for x in a.nig_seeds.split(",") if x != ""]
    load = lambda name: torch.load(CACHE.format(name), weights_only=False)
    # the protocol caches reuse the random split's calibration half (same tensors) but do not store its cycle ids
    rnd = load("random")
    out = {"random": s0(rnd)}
    for c in ["B0025", "B0026", "B0027", "B0028"]:
        out[f"protocol_{c}"] = s0(load(f"protocol_{c}"), rnd["cal_cycle"])
    for seed in seeds:
        for c in ["B0005", "B0006", "B0007", "B0018"]:
            out[f"loco_s{seed}_{c}"] = s0(load(f"loco_clean_s{seed}_{c}"))
    for k in nig:
        rk = load(f"random_s{k}")
        out[f"random_s{k}"] = s0(rk)
        for c in ["B0025", "B0026", "B0027", "B0028"]:
            pk = load(f"protocol_s{k}_{c}")
            assert torch.equal(pk["cal"]["soc"], rk["cal"]["soc"]), "protocol calibration differs from the random half"
            out[f"protocol_s{k}_{c}"] = s0(pk, rk["cal_cycle"])
    for k, v in out.items():
        a, z, p, g = v["all"], v["label_zero"], v["label_positive"], v["label_gt10"]
        print(f"{k:18s} zero-share {v['share_label_zero_pct']:5.1f}% | all RMSE {a['soc_rmse_pct']:6.2f} cov {a['s0_cov90_pct']:5.1f}"
              f" [{a['ci95_pct'][0]:5.1f},{a['ci95_pct'][1]:5.1f}]"
              f" | SOC>0: cov {p['s0_cov90_pct']:5.1f} | SOC>10%: cov {g.get('s0_cov90_pct', float('nan')):5.1f}"
              f" | SOH cov {v['soh_all']['s0_cov90_pct']:5.1f} [{v['soh_all']['ci95_pct'][0]:5.1f},{v['soh_all']['ci95_pct'][1]:5.1f}]")
    json.dump(out, open("results/revision/rest_windows.json", "w"), indent=1)
    print("wrote results/revision/rest_windows.json")


if __name__ == "__main__":
    main()

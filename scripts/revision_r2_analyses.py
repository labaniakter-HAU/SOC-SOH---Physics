"""Processes round 1, Reviewer 2: analyses of the saved held-out-cell and protocol predictions.

No model is trained or re-run; every number comes from results/revision/cache/ and uses the score,
quantile and density-ratio functions of the paper (uapi_former/shift_conformal.py).

  cal_size     R2.3   S0 coverage when the calibration partition is cut to 25 % or 50 % of its
                      cycles (20 random cycle subsets each), against the full partition.
  soh_matched  R2.4/5 target windows whose SOH label lies inside the calibration SOH range: S0
                      coverage inside and outside that range, and S1 refitted on the matched
                      windows only (an oracle selection, since it uses the labels).
  per_cycle    R2.6   S0 coverage per discharge cycle ("per trip"): share of cycles at >= 90 %.
  recal        R2.10  calibration on labelled cycles of the deployed cell itself, evaluated on the
                      same later cycles (rank >= 10): once on its first k cycles, or rolling on
                      the k cycles before each evaluated cycle (labels of a cycle become known when
                      it ends). Coverage and mean full width (percentage points), against S0.
  ev           R2.1   Stanford EV-profile cells (scripts/revision_conformal_clean.py --level ev):
                      zero-shot accuracy, S0 per cell, and the same recalibration per cell.

Reads  results/revision/cache/{loco_clean_s{seed}_B00xx,protocol_s{seed}_B00xx,ev_s{seed}}.pt
Writes results/revision/r2_analyses.json
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from uapi_former.shift_conformal import logistic_density_ratio, nonconformity, vanilla_q, weighted_q  # noqa: E402

CELLS = ["B0005", "B0006", "B0007", "B0018"]
PROT = ["B0025", "B0026", "B0027", "B0028"]
SEEDS = range(5)
NOM = 0.90
KEYS = {"soc": ("sg", "sv", "sa", "sb", "soc"), "soh": ("hg", "hv", "ha", "hb", "soh")}
FIRST_EVAL = 10  # evaluated cycles: rank >= 10, identical for every calibration variant


def score(d, task):
    return nonconformity(*(d[k] for k in KEYS[task]))


def pct(m):
    return 100 * float(m.float().mean())


def recal(s, sig, rank, s0_hit, s0_q):
    """one-off (first k cycles) and rolling (k previous cycles) calibration on the deployed cell"""
    ev = rank >= FIRST_EVAL
    out = {"n_eval_windows": int(ev.sum()), "S0": {"cov": pct(s0_hit[ev]), "width": float((2 * s0_q * sig[ev]).mean() * 100)}}
    for k in (1, 2, 5, 10):
        q = vanilla_q(s[rank < k], NOM)
        out[f"first{k}"] = {"cov": pct(s[ev] <= q), "width": float((2 * q * sig[ev]).mean() * 100)}
    for k in (3, 5, 10):
        hit, wid = [], []
        for r in range(FIRST_EVAL, int(rank.max()) + 1):
            q = vanilla_q(s[(rank >= r - k) & (rank < r)], NOM)
            m = rank == r
            hit.append(s[m] <= q)
            wid.append(2 * q * sig[m] * 100)
        out[f"rolling{k}"] = {"cov": pct(torch.cat(hit)), "width": float(torch.cat(wid).mean())}
    return out


def cycle_rank(cycles):
    c = np.array([int(str(x).split(":")[1]) for x in cycles])
    return torch.tensor(np.searchsorted(np.unique(c), c))


def held_out(seed, cell):
    d = torch.load(f"results/revision/cache/loco_clean_s{seed}_{cell}.pt", map_location="cpu", weights_only=False)
    c, t = d["cal"], d["tgt"]
    rank = cycle_rank(d["tgt_cycle"])
    cal_cyc = np.array([str(x) for x in d["cal_cycle"]])
    ucal = np.unique(cal_cyc)
    match = (t["soh"] >= c["soh"].min()) & (t["soh"] <= c["soh"].max())
    # one discriminator per run, shared by both tasks (as in the paper)
    w_cal = logistic_density_ratio(c["rep"], t["rep"][match])
    w_tgt = logistic_density_ratio(c["rep"], t["rep"][match], query=t["rep"][match])
    e = {"n_cal_cycles": len(ucal), "n_cal_windows": len(cal_cyc), "n_tgt_cycles": int(rank.max()) + 1,
         "share_soh_matched": pct(match)}
    gen = np.random.default_rng(seed)
    for task in ("soc", "soh"):
        s_c, _ = score(c, task)
        s_t, sig = score(t, task)
        q0 = vanilla_q(s_c, NOM)
        hit = s_t <= q0
        r = {"S0_all": pct(hit), "S0_matched": pct(hit[match]),
             "S0_unmatched": pct(hit[~match]) if (~match).any() else None}
        q1 = weighted_q(s_c, w_cal, NOM, w_test=w_tgt)
        acc = torch.isfinite(q1)
        r["S1_matched"] = {"accepted_cov": pct(s_t[match][acc] <= q1[acc]) if acc.any() else None,
                           "refusal": pct(~acc)}
        sizes = {}
        for frac in (0.25, 0.5):
            v = []
            for _ in range(20):
                keep = torch.tensor(np.isin(cal_cyc, gen.choice(ucal, max(2, round(frac * len(ucal))), replace=False)))
                v.append(pct(s_t <= vanilla_q(s_c[keep], NOM)))
            sizes[str(frac)] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}
        r["cal_size"] = sizes
        per = torch.stack([hit[rank == k].float().mean() for k in range(int(rank.max()) + 1)])
        r["per_cycle"] = {"share_ge90": pct(per >= NOM), "median": 100 * float(per.median())}
        r["recal"] = recal(s_t, sig, rank, hit, q0)
        e[task] = r
    return e


def protocol(seed, cell):
    d = torch.load(f"results/revision/cache/protocol_s{seed}_{cell}.pt", map_location="cpu", weights_only=False)
    c, t = d["cal"], d["tgt"]
    rank = cycle_rank(d["tgt_cycle"])
    e = {"n_tgt_cycles": int(rank.max()) + 1}
    for task in ("soc", "soh"):
        s_c, _ = score(c, task)
        s_t, sig = score(t, task)
        q0 = vanilla_q(s_c, NOM)
        e[task] = {"S0_all": pct(s_t <= q0), "recal": recal(s_t, sig, rank, s_t <= q0, q0)}
    return e


def ev(seed):
    """Stanford EV-profile cells (R2.1, zero-shot v12-NIG checkpoint): accuracy, S0, and the same
    recalibration on each cell's own labelled cycles."""
    d = torch.load(f"results/revision/cache/ev_s{seed}.pt", map_location="cpu", weights_only=False)
    c, t = d["cal"], d["tgt"]
    cell_of = np.array([str(x).split(":")[0] for x in d["tgt_cycle"]])
    rmse = lambda g, y, m=slice(None): float(100 * torch.sqrt(((t[g][m] - t[y][m]) ** 2).mean()))
    e = {"soc_rmse": rmse("sg", "soc"), "soh_rmse": rmse("hg", "soh"),
         "soc_r": float(np.corrcoef(t["sg"].numpy(), t["soc"].numpy())[0, 1]), "n_windows": len(cell_of),
         "cells": {}}
    for cell in sorted(set(cell_of)):
        m = torch.tensor(cell_of == cell)
        e["cells"][cell] = {"n_windows": int(m.sum()), "n_cycles": int(cycle_rank(d["tgt_cycle"][cell_of == cell]).max()) + 1,
                            "soh_min": float(t["soh"][m].min()), "soh_max": float(t["soh"][m].max()),
                            "soc_rmse": rmse("sg", "soc", m), "soh_rmse": rmse("hg", "soh", m)}
    for task in ("soc", "soh"):
        s_c, sig_c = score(c, task)
        s_t, sig = score(t, task)
        q0 = vanilla_q(s_c, NOM)
        # the evidential scale on the deployed windows relative to the calibration windows
        e[task] = {"S0_all": pct(s_t <= q0), "sigma_ratio": float(sig.median() / sig_c.median()), "cells": {}}
        for cell in sorted(set(cell_of)):
            m = torch.tensor(cell_of == cell)
            e[task]["cells"][cell] = {"S0": pct(s_t[m] <= q0),
                                      "recal": recal(s_t[m], sig[m], cycle_rank(d["tgt_cycle"][cell_of == cell]),
                                                     s_t[m] <= q0, q0)}
    return e


def main():
    torch.set_num_threads(4)
    out = {}
    for seed in SEEDS:
        out[f"loco_clean_s{seed}"] = {cell: held_out(seed, cell) for cell in CELLS}
        out[f"protocol_s{seed}"] = {cell: protocol(seed, cell) for cell in PROT}
        out[f"ev_s{seed}"] = ev(seed)
        print("seed", seed, {c: round(v["soc"]["recal"]["rolling5"]["cov"], 1) for c, v in out[f"loco_clean_s{seed}"].items()},
              flush=True)
    # self-check: S0 on all windows reproduces results/revision/rest_windows.json (up to float32 summation order)
    rw = json.load(open("results/revision/rest_windows.json"))
    for seed in SEEDS:
        for cell in CELLS:
            ref = rw[f"loco_s{seed}_{cell}"]
            assert abs(out[f"loco_clean_s{seed}"][cell]["soc"]["S0_all"] - ref["all"]["s0_cov90_pct"]) < 1e-4, (seed, cell)
            assert abs(out[f"loco_clean_s{seed}"][cell]["soh"]["S0_all"] - ref["soh_all"]["s0_cov90_pct"]) < 1e-4, (seed, cell)
    json.dump(out, open("results/revision/r2_analyses.json", "w"), indent=1)
    print("wrote results/revision/r2_analyses.json")


if __name__ == "__main__":
    main()

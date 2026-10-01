"""SP4: cross-protocol accuracy (Table S9) for the five v12-NIG seed checkpoints.

Computed from the conformal-study caches (results/revision/cache/protocol_s{k}_<cell>.pt, written
by scripts/revision_conformal_clean.py --level cross_protocol --nig-seed k). The same computation
on the canonical caches (protocol_<cell>.pt) must first reproduce results/cross_protocol_results.json
(scripts/cross_protocol_eval.py) to 1e-4, so the per-seed values use the identical metric.

Writes results/revision/seeds/cross_protocol_acc.json
"""
import json
import os

import numpy as np
import torch

CELLS = ("B0025", "B0026", "B0027", "B0028")


def acc(name):
    d = torch.load(f"results/revision/cache/{name}.pt", weights_only=False)["tgt"]
    s, sg, h, hg = (d[k].numpy().astype(np.float64) for k in ("soc", "sg", "soh", "hg"))
    return {"soc_rmse": float(100 * np.sqrt(np.mean((sg - s) ** 2))), "soh_rmse": float(100 * np.sqrt(np.mean((hg - h) ** 2))),
            "soc_r": float(np.corrcoef(sg, s)[0, 1]), "soh_r": float(np.corrcoef(hg, h)[0, 1]), "n": int(len(s))}


def main():
    ref = json.load(open("results/cross_protocol_results.json"))["cells"]
    for c in CELLS:
        a = acc(f"protocol_{c}")
        for k in ("soc_rmse", "soh_rmse", "soc_r", "soh_r"):
            assert abs(a[k] - ref[c][k]) < 1e-4, (c, k, a[k], ref[c][k])
    print("canonical cross_protocol_results.json reproduced from the caches")
    out = {f"s{k}": {c: acc(f"protocol_s{k}_{c}") for c in CELLS} for k in range(5)}
    os.makedirs("results/revision/seeds", exist_ok=True)
    json.dump(out, open("results/revision/seeds/cross_protocol_acc.json", "w"), indent=1)
    for k, v in out.items():
        print(k, {c: round(e["soc_rmse"], 2) for c, e in v.items()})
    print("wrote results/revision/seeds/cross_protocol_acc.json")


if __name__ == "__main__":
    main()

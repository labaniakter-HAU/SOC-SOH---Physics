"""SP4: fine-tuned test statistics per backbone seed for the cross-chemistry text.

For backbone seed s (scripts/revision_sp4_paths.py), on CPU:
  - Oxford: SOC/SOH RMSE and correlations of the fine-tuned model on the relabelled test split
    (must reproduce its evaluation log), split sizes (same content as the former
    results/revision/oxford_ft_stats.json);
  - MIT-TRI: per-test-cell SOC RMSE, window count and share of the aggregate squared SOC error of
    the fine-tuned model, computed from predictions (the former
    results/mit_tri_cell37_diagnostic.json held typed-in per-cell values); the aggregate must
    reproduce the evaluation log.

Writes results/revision/seeds/oxford_ft_stats_s{s}.json and mit_tri_percell_s{s}.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uapi_former.dataset import MITTRIDataset, OxfordDataset  # noqa: E402
from shift_conformal_study import load_model, collect  # noqa: E402
import revision_sp4_paths as sp  # noqa: E402

DEV = torch.device("cpu")


def logged(path, key):
    return float(re.findall(key + r"\s+RMSE\s*:\s*([0-9.]+)", open(path, encoding="utf-8", errors="replace").read())[-1])


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path + ".tmp", "w"), indent=2)
    os.replace(path + ".tmp", path)
    print("wrote", path)


def oxford(s):
    ds = OxfordDataset("data/raw/oxford", split="test", seq_len=200)
    model, meta = load_model(f"{sp.ft_dir('oxford', s)}/best.pt", DEV)
    r = collect(model, DataLoader(ds, batch_size=256, shuffle=False), DEV, meta)
    sp_, st = r["sg"].numpy(), r["soc"].numpy()
    hp, ht = r["hg"].numpy(), r["soh"].numpy()
    out = {"seed": s, "n_test": int(len(ds)),
           "soc_rmse_pct": float(100 * np.sqrt(np.mean((sp_ - st) ** 2))),
           "soh_rmse_pct": float(100 * np.sqrt(np.mean((hp - ht) ** 2))),
           "soc_r": float(np.corrcoef(sp_, st)[0, 1]), "soh_r": float(np.corrcoef(hp, ht)[0, 1]),
           "record": ds.record, "n_test_cells": len(set(ds.sample_cell)),
           "n_test_cycles": len({m[1] for m in ds.meta}),
           "n_train": len(OxfordDataset("data/raw/oxford", split="train", seq_len=200)),
           "n_val": len(OxfordDataset("data/raw/oxford", split="val", seq_len=200))}
    log = sp.ft_log("oxford", s)
    assert abs(out["soc_rmse_pct"] / 100 - logged(log, "SOC")) < 6e-5, (out["soc_rmse_pct"], log)
    assert abs(out["soh_rmse_pct"] / 100 - logged(log, "SOH")) < 6e-5, (out["soh_rmse_pct"], log)
    write(sp.oxford_stats_json(s), out)


def mit(s):
    ds = MITTRIDataset("data/raw/mit_tri", split="test")
    model, meta = load_model(f"{sp.ft_dir('mit_tri', s)}/best.pt", DEV)
    r = collect(model, DataLoader(ds, batch_size=256, shuffle=False), DEV, meta)
    err2 = ((r["sg"] - r["soc"]) ** 2).numpy()
    cells = np.array(ds.sample_cell)
    assert len(cells) == len(err2)
    agg = float(100 * np.sqrt(err2.mean()))
    assert abs(agg / 100 - logged(sp.ft_log("mit_tri", s), "SOC")) < 6e-5, (agg, s)
    per = {}
    for c in sorted(set(cells.tolist())):
        m = cells == c
        per[c] = {"n_windows": int(m.sum()), "soc_rmse_pct": float(100 * np.sqrt(err2[m].mean())),
                  "share_of_squared_error": float(err2[m].sum() / err2.sum()),
                  "share_of_windows": float(m.mean())}
    write(sp.mit_diag_json(s), {"seed": s, "aggregate_soc_rmse_pct": agg, "n_cells": len(per), "per_cell": per})


def check_canonical():
    """The per-cell computation must reproduce the typed per-cell values of the former single
    lineage (results/mit_tri_cell37_diagnostic.json, two decimals) before it is used per seed."""
    import tempfile
    typed = json.load(open("results/mit_tri_cell37_diagnostic.json"))["per_cell_test_rmse"]
    global write
    keep, got = write, {}
    write = lambda path, obj: got.update(obj)
    try:
        mit(None)
    finally:
        write = keep
    mine = sorted((v["n_windows"], round(v["soc_rmse_pct"], 2)) for v in got["per_cell"].values())
    ref = sorted((v["n_windows"], v["soc_rmse"]) for v in typed.values())
    print("computed:", mine, "\ntyped:   ", ref)
    assert mine == ref, "per-cell computation does not reproduce the typed diagnostic"
    print("canonical per-cell values reproduced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check-canonical", action="store_true")
    a = ap.parse_args()
    if a.check_canonical:
        check_canonical()
        return
    oxford(a.seed)
    mit(a.seed)


if __name__ == "__main__":
    main()

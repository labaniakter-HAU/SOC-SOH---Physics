"""Oxford fine-tuned test statistics quoted in the cross-chemistry section (SP2: relabelled loader).

Recomputes, from checkpoints/oxford_ft/best.pt on the relabelled (1C discharge record) Oxford
test split on CPU: SOC/SOH RMSE (which must reproduce logs/revision/sp2/ox_ft_eval.log), the
SOC and SOH correlations, and the split sizes quoted in Table 1 and Sec. 4.10.

Writes results/revision/oxford_ft_stats.json
"""
import json, os, re, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uapi_former.dataset import OxfordDataset
from shift_conformal_study import load_model, collect

dev = torch.device("cpu")
ds = OxfordDataset("data/raw/oxford", split="test", seq_len=200)
model, meta = load_model("checkpoints/oxford_ft/best.pt", dev)
r = collect(model, DataLoader(ds, batch_size=256, shuffle=False), dev, meta)
soc_p, soc_t = r["sg"].numpy(), r["soc"].numpy()
soh_p, soh_t = r["hg"].numpy(), r["soh"].numpy()
out = {"n": int(len(ds)), "n_test": int(len(ds)),
       "soc_rmse_pct": float(100 * np.sqrt(np.mean((soc_p - soc_t) ** 2))),
       "soh_rmse_pct": float(100 * np.sqrt(np.mean((soh_p - soh_t) ** 2))),
       "soh_r": float(np.corrcoef(soh_p, soh_t)[0, 1]),
       "soc_r": float(np.corrcoef(soc_p, soc_t)[0, 1]),
       "soc_target_sd_pct": float(100 * np.std(soc_t, ddof=1)),
       "soh_target_sd_pct": float(100 * np.std(soh_t, ddof=1))}
out.update({"record": ds.record, "n_test_cells": len(set(ds.sample_cell)),
            "n_test_cycles": len({m[1] for m in ds.meta}),
            "n_train": len(OxfordDataset("data/raw/oxford", split="train", seq_len=200)),
            "n_val": len(OxfordDataset("data/raw/oxford", split="val", seq_len=200))})
log = open("logs/revision/sp2/ox_ft_eval.log", encoding="utf-8", errors="replace").read()
soc_log = float(re.findall(r"SOC\s+RMSE\s*:\s*([0-9.]+)", log)[-1])
soh_log = float(re.findall(r"SOH\s+RMSE\s*:\s*([0-9.]+)", log)[-1])
assert abs(out["soc_rmse_pct"] / 100 - soc_log) < 6e-5 and abs(out["soh_rmse_pct"] / 100 - soh_log) < 6e-5, (out, soc_log, soh_log)
with open("results/revision/oxford_ft_stats.json.tmp", "w") as f:
    json.dump(out, f, indent=2)
os.replace("results/revision/oxford_ft_stats.json.tmp", "results/revision/oxford_ft_stats.json")
print(json.dumps(out, indent=2))

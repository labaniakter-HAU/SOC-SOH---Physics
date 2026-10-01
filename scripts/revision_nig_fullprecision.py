"""Full-precision test RMSE for the five-seed v12-NIG campaign.

results/multiseed_v12_nig.json stores each seed's RMSE rounded to 0.01 percentage
points (it was transcribed from logs/_nig_multiseed/seed*_eval.txt), while the
v12-MSE campaign (results/_multiseed_raw/full.json) keeps four decimals. Table 2
prints three decimals and the NIG-vs-MSE paired t-test was run on the mixed
precisions. This script re-evaluates the five checkpoints on the same test split,
checks each value rounds to the logged one, and redoes the paired test on
full-precision values for both campaigns.

Writes results/revision/nig_multiseed_fullprecision.json
"""
import json, os, sys
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uapi_former.dataset import NASABatteryDataset
from shift_conformal_study import load_model, collect

dev = torch.device("cpu")          # fp32, bit-reproducible
ds = NASABatteryDataset("data/raw/nasa", split="test", split_mode="intra_cell_random")
ld = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
logged = {r["seed"]: r for r in json.load(open("results/multiseed_v12_nig.json"))["seeds"]}

out = {"source": "scripts/revision_nig_fullprecision.py", "n_test": len(ds), "seeds": []}
for s in range(5):
    ck = f"checkpoints/nasa_v12_nig_seed{s}/best.pt"
    model, meta = load_model(ck, dev)
    r = collect(model, ld, dev, meta)
    soc = 100 * float(torch.sqrt(((r["sg"] - r["soc"]) ** 2).mean()))
    soh = 100 * float(torch.sqrt(((r["hg"] - r["soh"]) ** 2).mean()))
    epoch = torch.load(ck, map_location="cpu", weights_only=False)["epoch"]
    assert round(soc, 2) == logged[s]["soc_rmse"] and round(soh, 2) == logged[s]["soh_rmse"], \
        (s, soc, soh, logged[s])
    out["seeds"].append({"seed": s, "soc_rmse": soc, "soh_rmse": soh, "best_epoch": epoch})
    print(f"seed {s}: SOC {soc:.4f}  SOH {soh:.4f}  (logged {logged[s]['soc_rmse']}/{logged[s]['soh_rmse']}, epoch {epoch})", flush=True)

mse = json.load(open("results/_multiseed_raw/full.json"))
mse = mse if isinstance(mse, list) else (mse.get("runs") or mse.get("seeds"))
mse = {e["seed"]: e for e in mse}
for t in ("soc", "soh"):
    a = [100 * mse[s][f"{t}_rmse"] for s in range(5)]
    b = [e[f"{t}_rmse"] for e in out["seeds"]]
    res = stats.ttest_rel(b, a)
    out[f"paired_ttest_vs_v12_mse_{t}"] = {"t": float(res.statistic), "p": float(res.pvalue)}
    print(f"{t}: NIG {np.mean(b):.4f} +- {np.std(b, ddof=1):.4f}  MSE {np.mean(a):.4f} +- {np.std(a, ddof=1):.4f}  paired p = {res.pvalue:.4f}")
json.dump(out, open("results/revision/nig_multiseed_fullprecision.json", "w"), indent=2)
print("wrote results/revision/nig_multiseed_fullprecision.json")

"""Fit per-cell DC internal resistance R0 from raw discharge data and test
whether it explains/improves on the fixed R0=0.15 Ohm assumption used
throughout UAPI-Former's EITE feature construction (addresses the
"Fixed internal-resistance assumption" limitation with real evidence rather
than only synthetic +/-X% perturbations).

Method: for each cell, using its own discharge cycles, estimate SOC via
Coulomb counting (SOC_t = 1 - cumulative_Ah_t / cycle_capacity), then fit
R0 by closed-form weighted least squares on
    OCV(SOC_t) - V_t = I_t * R0 + noise
i.e. R0_hat = sum(I_t * e_t) / sum(I_t^2), e_t = OCV(SOC_t) - V_t.

Then, for each cell's fitted R0, re-evaluate the frozen canonical checkpoint
(no retraining) on the full NASA intra_cell_random test set with that R0
substituted globally into V_ocv_norm / soc_from_ocv feature construction --
the same global-substitution technique as the existing R0 sensitivity sweep
(results/r0_sensitivity.json), but now anchored to real fitted values
instead of arbitrary perturbations.

Usage: python scripts/fit_r0_per_cell.py
Writes results/r0_per_cell.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from uapi_former.baselines_ekf import load_discharge_cycles, ocv_nmc
from uapi_former.dataset import NASABatteryDataset
from uapi_former.model import UAPIFormer

CELLS = ["B0005", "B0006", "B0007", "B0018"]
DATA_DIR = "data/raw/nasa"
CKPT = "checkpoints/nasa_v12/best.pt"


def fit_r0(cell):
    cycles = load_discharge_cycles(os.path.join(DATA_DIR, f"{cell}.mat"), cell)
    num, den = 0.0, 0.0
    n_pts = 0
    for c in cycles:
        if not np.isfinite(c["capacity"]) or len(c["V"]) < 20:
            continue
        dt = np.diff(c["t"], prepend=c["t"][0])
        i_a = np.abs(c["I"])
        q_ah = np.cumsum(i_a * dt) / 3600.0
        soc_t = np.clip(1.0 - q_ah / c["capacity"], 0.0, 1.0)
        e_t = ocv_nmc(soc_t) - c["V"]          # = I*R0 + noise
        num += float(np.sum(i_a * e_t))
        den += float(np.sum(i_a ** 2))
        n_pts += len(i_a)
    r0_hat = num / den if den > 0 else float("nan")
    return r0_hat, n_pts


@torch.no_grad()
def eval_with_r0_classpatch(model, device, r0_value):
    """R0 is read at construction time inside _load_cell_random_split via
    self.R0, so it must be patched on the CLASS default before instantiation
    (setting the attribute on an already-built instance has no effect)."""
    orig = NASABatteryDataset.R0
    NASABatteryDataset.R0 = r0_value
    try:
        ds = NASABatteryDataset(DATA_DIR, split="test", split_mode="intra_cell_random")
    finally:
        NASABatteryDataset.R0 = orig
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    soc_se, soh_se, n = 0.0, 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch; v_spme = None
        else:
            x, v_spme, soc, soh = batch; v_spme = v_spme.to(device)
        x = x.to(device)
        (sg, *_), (hg, *_) = model(x, v_spme)
        soc_se += float(((sg.cpu() - soc.view(-1)) ** 2).sum())
        soh_se += float(((hg.cpu() - soh.view(-1)) ** 2).sum())
        n += x.size(0)
    return {"soc_rmse": float(np.sqrt(soc_se / n)) * 100, "soh_rmse": float(np.sqrt(soh_se / n)) * 100}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location="cpu")
    a = ckpt.get("args", {})
    model = UAPIFormer(in_channels=a.get("in_channels", 6), d_model=a.get("d_model", 128),
                       nhead=a.get("nhead", 4), num_layers=a.get("num_layers", 4),
                       seq_len=a.get("seq_len", 200))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    out = {"fixed_r0": 0.15, "per_cell_fit": {}, "eval_at_fitted_r0": {}}
    for cell in CELLS:
        r0_hat, n_pts = fit_r0(cell)
        out["per_cell_fit"][cell] = {"r0_hat": r0_hat, "n_points": n_pts,
                                     "pct_diff_from_fixed": (r0_hat - 0.15) / 0.15 * 100}
        print(f"{cell}: R0_hat={r0_hat:.4f} Ohm ({(r0_hat-0.15)/0.15*100:+.1f}% vs fixed 0.15), n={n_pts}")

    r0_values = [out["per_cell_fit"][c]["r0_hat"] for c in CELLS]
    mean_r0 = float(np.mean(r0_values))
    out["mean_fitted_r0"] = mean_r0
    out["std_fitted_r0"] = float(np.std(r0_values, ddof=1))
    print(f"\nMean fitted R0={mean_r0:.4f} +/- {out['std_fitted_r0']:.4f} Ohm "
         f"(fixed value used in training: 0.15 Ohm)")

    print("\nRe-evaluating frozen checkpoint (zero-shot, no retraining) at each "
         "cell's fitted R0 and at the dataset-mean fitted R0:")
    for cell in CELLS + ["mean"]:
        r0v = mean_r0 if cell == "mean" else out["per_cell_fit"][cell]["r0_hat"]
        m = eval_with_r0_classpatch(model, device, r0v)
        out["eval_at_fitted_r0"][cell] = {"r0_used": r0v, **m}
        print(f"  R0({cell})={r0v:.4f}: SOC RMSE={m['soc_rmse']:.2f}%  SOH RMSE={m['soh_rmse']:.2f}%")

    out_path = "results/r0_per_cell.json"
    assert not os.path.exists(out_path), f"refusing to overwrite {out_path}"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

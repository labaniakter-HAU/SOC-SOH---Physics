"""Revision R1.6 (second half): baselines under CROSS-PROTOCOL shift.

The reviewer asked for the baselines to be carried into the LOCO *and* the
cross-protocol settings. LOCO is handled by scripts/revision_baseline_conformal.py.
This script covers cross-protocol, with every estimator on an identical protocol:

  * models    : UAPI-Former v12-NIG five-seed campaign (checkpoints/nasa_v12_nig_seed{s})
                and the two baselines retrained under the same protocol
                (checkpoints/baseline_{name}_seed{s}_random{tag}): intra-cell random
                split, seeds 0-4, AdamW 1e-3 / wd 1e-4 / cosine, batch 256,
                400 epochs fixed, best-validation checkpoint.
  * calibration: the random half of the random-split TEST cycles (seed 2026),
                 exactly as the cross_protocol level of revision_conformal_clean.py;
                 one mask, built once, shared by all three estimators.
  * targets   : B0025-B0028 (lower-current protocol), all windows.
  * score     : absolute residual (sigma == 1), because the baselines emit no scale.
  * S1        : density-ratio weights in each estimator's SHARED representation
                (UAPI-Former: pooled latent; baselines: input of the SOC head).

Writes results/revision/baseline_protocol.json  {seed: {cell: {estimator: {...}}}}
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uapi_former import shift_conformal as sc
from uapi_former.dataset import NASABatteryDataset
from scripts.shift_conformal_study import load_model, collect
from revision_baselines import make, MODELS
from revision_baseline_conformal import collect_baseline, metrics, s1_diagnostics
from train_baseline import ChannelSubsetDataset

COV, CAL_SEED = 0.90, 2026
TARGETS = ["B0025", "B0026", "B0027", "B0028"]
mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def calibration_mask(ds):
    """Identical construction to revision_conformal_clean.clean_random_halves."""
    keys = np.array([f"{c}:{p}" for c, p, _ in ds.meta])
    ukeys = sorted(set(keys.tolist()))
    perm = np.random.RandomState(CAL_SEED).permutation(len(ukeys))
    cal_keys = set(ukeys[k] for k in perm[: len(ukeys) // 2])
    return np.array([k in cal_keys for k in keys])


def schemes(y_cal, g_cal, r_cal, y_t, g_t, r_t):
    s_cal = torch.abs(y_cal - g_cal)
    q0 = sc.vanilla_q(s_cal, COV)
    out = {"S0": metrics(y_t, g_t, q0),
           "rmse_pct": float(100 * torch.sqrt(((y_t - g_t) ** 2).mean()))}
    w_cal = sc.logistic_density_ratio(r_cal, r_t)
    w_t = sc.logistic_density_ratio(r_cal, r_t, query=r_t)
    q1 = sc.weighted_q(s_cal, w_cal, COV, w_test=w_t)
    out["S1"] = metrics(y_t, g_t, q1)
    out["S1"].update(s1_diagnostics(r_cal, r_t, w_cal))
    # SOC > 0 stratum (R1.4): about half of the B0025-B0027 windows are end-of-discharge /
    # rest windows labelled SOC = 0, which every estimator predicts almost exactly; pooled
    # coverage there is dominated by those windows. Same quantiles, restricted evaluation.
    pos = y_t > 1e-6
    q1p = q1[pos] if torch.is_tensor(q1) and q1.dim() > 0 else q1
    out["share_soc_zero_pct"] = float(100 * (~pos).float().mean())
    out["rmse_pct_soc_pos"] = float(100 * torch.sqrt(((y_t[pos] - g_t[pos]) ** 2).mean()))
    out["S0_soc_pos"] = metrics(y_t[pos], g_t[pos], q0)
    out["S1_soc_pos"] = metrics(y_t[pos], g_t[pos], q1p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cells", nargs="+", default=TARGETS)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--ckpt-tag", default="_fixed")
    ap.add_argument("--uapi-ckpt", default="checkpoints/nasa_v12_nig_seed{seed}/best.pt")
    ap.add_argument("--out", default="results/revision/baseline_protocol.json")
    # CPU by default: cuDNN's TF32 convolutions change the baseline representations by
    # up to ~0.09 and the density-ratio fit turns that into ~1 point of S1 coverage.
    # CPU fp32 is bit-reproducible and needs no GPU; every seed must use one device.
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    # Datasets and the calibration mask are built ONCE and shared by every estimator.
    test_ds = NASABatteryDataset("data/raw/nasa", split="test", split_mode="intra_cell_random")
    is_cal = torch.as_tensor(calibration_mask(test_ds))
    tgt_ds = {c: NASABatteryDataset("data/raw/nasa", cells=[c]) for c in a.cells}
    print(f"calibration windows: {int(is_cal.sum())} of {len(is_cal)} test windows "
          f"({len(set(k for k, m in zip([f'{c}:{p}' for c, p, _ in test_ds.meta], is_cal) if m))} cycles)",
          flush=True)

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for seed in a.seeds:
        rs = res.setdefault(str(seed), {})

        # ---- UAPI-Former -------------------------------------------------------
        model, meta = load_model(a.uapi_ckpt.format(seed=seed), dev)
        full = collect(model, mk(test_ds), dev, meta)
        cal = {k: v[is_cal] for k, v in full.items()}
        for cell in a.cells:
            t = collect(model, mk(tgt_ds[cell]), dev, meta)
            rs.setdefault(cell, {})["uapi_former"] = schemes(
                cal["soc"], cal["sg"], cal["rep"], t["soc"], t["sg"], t["rep"])
        del model

        # ---- baselines -----------------------------------------------------------
        for name in a.models:
            ck = f"checkpoints/baseline_{name}_seed{seed}_random{a.ckpt_tag}/best.pt"
            if not os.path.exists(ck):
                print("missing", ck, flush=True)
                continue
            raw = torch.load(ck, map_location="cpu", weights_only=False)
            ch = raw["in_channels"]
            m = make(name, ch).to(dev)
            m.load_state_dict(raw["model_state"]); m.eval()
            g_all, y_all, r_all = collect_baseline(m, mk(ChannelSubsetDataset(test_ds, n_channels=ch)), dev)
            g_c, y_c, r_c = g_all[is_cal], y_all[is_cal], r_all[is_cal]
            for cell in a.cells:
                g_t, y_t, r_t = collect_baseline(m, mk(ChannelSubsetDataset(tgt_ds[cell], n_channels=ch)), dev)
                rs.setdefault(cell, {})[name] = schemes(y_c, g_c, r_c, y_t, g_t, r_t)

        for cell in a.cells:
            row = rs[cell]
            print(f"seed{seed} {cell}", {k: (round(v['rmse_pct'], 2), round(100 * v['S0']['coverage'], 1),
                                              round(100 * v['S1']['refusal'], 1)) for k, v in row.items()},
                  flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

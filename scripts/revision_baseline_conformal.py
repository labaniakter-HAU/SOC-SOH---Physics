"""Revision R1.6: is the coverage collapse a property of the estimator or of the
calibration scheme?

The baselines produce a point estimate but no scale, so the sigma-normalized
score of the main text is not available. Split conformal prediction does not
need one: with sigma == 1 the score is the plain absolute residual and the
interval is a constant-width band. We therefore run, for every clean LOCO fold
and for each estimator (UAPI-Former, CNN-BiLSTM, PI-Transformer):

  S0   split conformal with the absolute-residual score, calibrated on the
       fold's dedicated calibration partition;
  S1   the same score with density-ratio weights estimated in that estimator's
       OWN penultimate representation (transductive regime, as in the main text).

If coverage collapses for all three estimators, the failure belongs to the
source-calibrated scheme rather than to UAPI-Former.

Writes results/revision/baseline_conformal_s{seed}.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uapi_former import shift_conformal as sc
from uapi_former.dataset import NASABatteryDataset
from revision_baselines import loaders_loco, make, MODELS, CELLS
from uapi_former.conformal_extras import active_mask, restrict_active

COV = 0.90


@torch.no_grad()
def collect_baseline(model, loader, dev):
    """Point predictions plus the SHARED representation that feeds both task heads.

    This is the analogue of UAPI-Former's 128-d pooled latent: the input of the
    first layer of the SOC head, which is the output of the shared trunk and is
    identical to the input of the SOH head. (An earlier version hooked the last
    nn.Linear in module order, which is the SOH head's private output layer.)"""
    feats = []
    target = model.soc_head[0]
    assert isinstance(target, torch.nn.Linear), "expected soc_head[0] to be the head's input Linear"
    h = target.register_forward_hook(lambda mod, inp, out: feats.append(inp[0].detach().cpu()))
    P, Y = [], []
    for x, soc, _ in loader:
        ps, _ = model(x.to(dev))
        P.append(ps.view(-1).cpu()); Y.append(soc.view(-1))
    h.remove()
    n = sum(len(p) for p in P)
    rep = torch.cat(feats)
    assert len(rep) == n, f"representation/prediction misalignment: {len(rep)} vs {n}"
    return torch.cat(P), torch.cat(Y), rep


def s1_diagnostics(r_cal, r_tgt, w_cal):
    """How far each estimator's representation separates calibration from target:
    in-sample discriminator AUC (0.5 = indistinguishable) and the effective sample
    size of the S1 calibration weights as a fraction of the calibration set.
    Same conventions as the support diagnostics of the main text."""
    return {"auc": sc.discriminator_auc(r_cal, r_tgt),
            "ess_frac": sc.effective_sample_size(w_cal) / len(w_cal)}


def metrics(y, g, q):
    if isinstance(q, torch.Tensor):
        finite = torch.isfinite(q)
        covered = torch.ones_like(q, dtype=torch.bool)
        covered[finite] = torch.abs(y - g)[finite] <= q[finite]
        r = float((~finite).float().mean())
        w = float((2 * q[finite]).mean()) if finite.any() else float("nan")
    else:
        covered = torch.abs(y - g) <= q
        r, w = 0.0, float(2 * q)
    cov = float(covered.float().mean())
    return {"coverage": cov, "refusal": r,
            "accepted_coverage": (cov - r) / (1 - r) if r < 1 else float("nan"),
            "finite_and_correct": cov - r, "mean_width_pp": 100 * w, "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    # CPU by default: cuDNN's TF32 convolutions change the baseline representations by
    # up to ~0.09 and the density-ratio fit turns that into ~1 point of S1 coverage.
    # CPU fp32 is bit-reproducible and needs no GPU; every seed must use one device.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-tag", default="",
                    help="suffix of the LOCO baseline checkpoint dirs (e.g. _fixed100)")
    ap.add_argument("--active", action="store_true",
                    help="calibrate and evaluate on windows with SOC label > 0 only (spec 2026-09-24 §4.1)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = {}
    for cell in CELLS:
        entry = {}
        # --- UAPI-Former reference, same score (absolute residual, sigma == 1) ---
        c = torch.load(f"results/revision/cache/loco_clean_s{a.seed}_{cell}.pt",
                       map_location="cpu", weights_only=False)
        cal, tgt = c["cal"], c["tgt"]
        if a.active:
            cal, tgt = restrict_active(cal, tgt)
        s_cal = torch.abs(cal["soc"] - cal["sg"])
        q0 = sc.vanilla_q(s_cal, COV)
        e = {"S0": metrics(tgt["soc"], tgt["sg"], q0)}
        w_cal = sc.logistic_density_ratio(cal["rep"], tgt["rep"])
        w_tgt = sc.logistic_density_ratio(cal["rep"], tgt["rep"], query=tgt["rep"])
        e["S1"] = metrics(tgt["soc"], tgt["sg"], sc.weighted_q(s_cal, w_cal, COV, w_test=w_tgt))
        e["S1"].update(s1_diagnostics(cal["rep"], tgt["rep"], w_cal))
        entry["uapi_former"] = e

        # --- learned baselines ------------------------------------------------
        for name in MODELS:
            ck = f"checkpoints/baseline_{name}_s{a.seed}_{cell}{a.ckpt_tag}/best.pt"
            if not os.path.exists(ck):
                print("missing", ck); continue
            raw = torch.load(ck, map_location="cpu", weights_only=False)
            ch = raw["in_channels"]
            model = make(name, ch).to(dev)
            model.load_state_dict(raw["model_state"]); model.eval()
            ld, _ = loaders_loco(cell, a.seed, ch, 256)
            gc_, yc, rc = collect_baseline(model, ld["cal"], dev)
            gt_, yt, rt = collect_baseline(model, ld["test"], dev)
            if a.active:
                mc, mt = active_mask(yc), active_mask(yt)
                gc_, yc, rc, gt_, yt, rt = gc_[mc], yc[mc], rc[mc], gt_[mt], yt[mt], rt[mt]
            s_c = torch.abs(yc - gc_)
            q = sc.vanilla_q(s_c, COV)
            e = {"S0": metrics(yt, gt_, q), "rmse_pct": float(100 * torch.sqrt(((yt - gt_) ** 2).mean()))}
            wc = sc.logistic_density_ratio(rc, rt)
            wt = sc.logistic_density_ratio(rc, rt, query=rt)
            e["S1"] = metrics(yt, gt_, sc.weighted_q(s_c, wc, COV, w_test=wt))
            e["S1"].update(s1_diagnostics(rc, rt, wc))
            entry[name] = e
        out[cell] = entry
        print(cell, {k: (round(100 * v["S0"]["coverage"], 1), round(100 * v["S1"]["accepted_coverage"], 1),
                         round(100 * v["S1"]["refusal"], 1)) for k, v in entry.items()}, flush=True)
    path = a.out or f"results/revision/baseline_conformal_s{a.seed}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

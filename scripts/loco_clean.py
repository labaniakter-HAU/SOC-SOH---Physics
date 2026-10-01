"""Clean LOCO protocol (revision, reviewer comments R2.2 + R2.8).

Differences from scripts/loco_cv.py (the submitted protocol):
  1. NMC OCV-SOC table is fitted per fold by the automated rule in
     uapi_former/ocv_fit.py, using ONLY the training-partition cycles of the
     three source cells (the held-out cell is never consulted).
  2. Q_cum is normalised by rated capacity (2 Ah), not by the measured
     first-cycle capacity of each cell.
  3. Source-cell cycles are split at cycle level into train / val / cal
     (85 / 7.5 / 7.5 %). val drives early stopping only; cal is used only for
     conformal calibration and is never seen by training or model selection.

Usage:
  py scripts/loco_clean.py --smoke --held-out B0006          # 1-epoch smoke
  py scripts/loco_clean.py --seed 0                          # 4 folds
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uapi_former.dataset import NASABatteryDataset
from uapi_former.model import UAPIFormer
from uapi_former.ocv_fit import fit_nmc_ocv_table, n_discharge_cycles
from scripts.loco_cv import train_epoch, _val_soc_rmse, evaluate, MODEL_KWARGS, CELLS


def fold_datasets(cell, data_dir, seed):
    src = [c for c in CELLS if c != cell]
    train_pos = {c: NASABatteryDataset.loco_random3_positions(
        c, n_discharge_cycles(data_dir, c), seed)["train"] for c in src}
    ocv_pts = fit_nmc_ocv_table(data_dir, train_pos, r0=NASABatteryDataset.R0)
    kw = dict(split_mode="loco", held_out_cell=cell, ocv_pts=ocv_pts,
              q_norm="rated", loco_split="random3", loco_seed=seed)
    ds = {s: NASABatteryDataset(data_dir, split=s, **kw) for s in ("train", "val", "cal", "test")}
    return ds, ocv_pts


def eval_only(cell, args, device):
    """Recompute fold metrics from an existing checkpoint (no training)."""
    ck = os.path.join("checkpoints", f"loco_clean_s{args.seed}_{cell}", "best.pt")
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    ds, ocv_pts = fold_datasets(cell, args.data_dir, args.seed)
    assert np.allclose(np.array(raw["ocv_pts"]), ocv_pts), "OCV table mismatch"
    model = UAPIFormer(**MODEL_KWARGS).to(device)
    model.load_state_dict(raw["model_state"])
    m = evaluate(model, DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False), device)
    m.update({"best_epoch": raw["epoch"], "val_soc_rmse": raw["val_soc_rmse"],
              "n_train": len(ds["train"]), "n_val": len(ds["val"]), "n_cal": len(ds["cal"]),
              "n_test": len(ds["test"]), "ocv_pts": ocv_pts.tolist(), "eval_only": True})
    print(f"  TEST {cell}: SOC={m['soc_rmse']:.3f}% SOH={m['soh_rmse']:.3f}% "
          f"r_soc={m['soc_r']:.4f} r_soh={m['soh_r']:.4f} (from checkpoint)", flush=True)
    return m


def run_fold(cell, args, device):
    if args.eval_only:
        return eval_only(cell, args, device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds, ocv_pts = fold_datasets(cell, args.data_dir, args.seed)
    print(f"\n=== clean fold held_out={cell} seed={args.seed} | "
          + " ".join(f"{k}={len(v)}" for k, v in ds.items()), flush=True)
    print("  fold OCV knots:", np.round(ocv_pts, 3).tolist(), flush=True)
    pin = device.type == "cuda"
    mk = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh,
                                  num_workers=0, pin_memory=pin)
    tr, va, te = mk(ds["train"], True), mk(ds["val"], False), mk(ds["test"], False)

    model = UAPIFormer(**MODEL_KWARGS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    tag = "smoke" if args.smoke else f"s{args.seed}"
    ckpt_dir = os.path.join("checkpoints", f"loco_clean_{tag}_{cell}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best, best_ep, best_state, patience, t0 = float("inf"), 0, None, 0, time.time()
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, tr, opt, scaler, device); sched.step()
        v_soc, v_soh = _val_soc_rmse(model, va, device)
        print(f"Epoch {ep}/{args.epochs} loss={loss:.4f} val_SOC={v_soc:.3f}% "
              f"val_SOH={v_soh:.3f}% ({time.time()-t0:.0f}s)", flush=True)
        if v_soc < best:
            best, best_ep, patience = v_soc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at {ep}", flush=True); break
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "epoch": best_ep, "val_soc_rmse": best,
                "args": MODEL_KWARGS, "ocv_pts": ocv_pts.tolist(), "q_norm": "rated",
                "loco_split": "random3", "loco_seed": args.seed, "held_out": cell},
               os.path.join(ckpt_dir, "best.pt"))
    m = evaluate(model, te, device)
    m.update({"best_epoch": best_ep, "val_soc_rmse": best,
              "n_train": len(ds["train"]), "n_val": len(ds["val"]),
              "n_cal": len(ds["cal"]), "n_test": len(ds["test"]),
              "ocv_pts": ocv_pts.tolist(), "train_seconds": time.time() - t0})
    print(f"  TEST {cell}: SOC={m['soc_rmse']:.3f}% SOH={m['soh_rmse']:.3f}% "
          f"r_soc={m['soc_r']:.4f} r_soh={m['soh_r']:.4f}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--held-out", default=None, choices=CELLS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default="data/raw/nasa")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="Recompute fold metrics from existing checkpoints instead of training.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1
    device = torch.device(args.device)
    cells = [args.held_out] if args.held_out else CELLS
    res = {c: run_fold(c, args, device) for c in cells}
    os.makedirs("results/revision", exist_ok=True)
    out = ("results/revision/_smoke_loco_clean.json" if args.smoke
           else f"results/revision/loco_clean_s{args.seed}.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(res)
    json.dump(prev, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()

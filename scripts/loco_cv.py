"""Leave-One-Cell-Out (LOCO) cross-validation for UAPI-Former.

Runs 4 folds over NASA cells [B0005, B0006, B0007, B0018].  Each fold holds
out one cell for testing, trains on the remaining three, and validates on a
cycle-fraction slice of those same cells.

Usage:
    # Run all 4 folds sequentially:
    python scripts/loco_cv.py --epochs 100

    # Run a single fold (e.g. for parallel GPU jobs):
    python scripts/loco_cv.py --held-out B0005 --epochs 100
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uapi_former.dataset import NASABatteryDataset
from uapi_former.evidential import nig_loss
from uapi_former.model import UAPIFormer

CELLS = ["B0005", "B0006", "B0007", "B0018"]

# Fixed model hyperparameters matching the canonical Paper B configuration.
MODEL_KWARGS = dict(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=200)

# NIG evidence regularization coefficient — same default as train.py.
NIG_COEFF = 0.1


# ---------------------------------------------------------------------------
# Training helpers (mirrored from train.py)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch
            v_spme = None
        else:
            x, v_spme, soc, soh = batch
            v_spme = v_spme.to(device)

        x   = x.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            (sg, sv, sa, sb), (hg, hv, ha, hb) = model(x, v_spme)
            # NIG evidential loss + MSE anchor to prevent lazy-mean collapse.
            loss = (nig_loss(soc, sg, sv, sa, sb, coeff=NIG_COEFF)
                    + nig_loss(soh, hg, hv, ha, hb, coeff=NIG_COEFF)
                    + 10.0 * F.mse_loss(sg, soc)
                    + 2.0  * F.mse_loss(hg, soh))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(1, n)


@torch.no_grad()
def _val_soc_rmse(model, loader, device):
    """Return SOC RMSE (%) and SOH RMSE (%) on the validation split.

    Uses the predicted NIG mean (γ) only — matches checkpoint selection in
    train.py which avoids monitoring the full ELBO to prevent NIG collapse.
    """
    model.eval()
    soc_se, soh_se, n = 0.0, 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch
            v_spme = None
        else:
            x, v_spme, soc, soh = batch
            v_spme = v_spme.to(device)
        x   = x.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)
        (sg, *_), (hg, *_) = model(x, v_spme)
        bs = x.size(0)
        soc_se += F.mse_loss(sg, soc, reduction="sum").item()
        soh_se += F.mse_loss(hg, soh, reduction="sum").item()
        n += bs
    model.train()
    soc_rmse = float(np.sqrt(soc_se / max(1, n))) * 100.0  # fraction → %
    soh_rmse = float(np.sqrt(soh_se / max(1, n))) * 100.0
    return soc_rmse, soh_rmse


# ---------------------------------------------------------------------------
# Evaluation metrics helper (specified in task)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Collect SOC/SOH predictions and return RMSE, Pearson r, sample count."""
    model.eval()
    soc_pred, soc_true, soh_pred, soh_true = [], [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch
            v_spme = None
        else:
            x, v_spme, soc, soh = batch
            v_spme = v_spme.to(device)
        x = x.to(device)
        (sg, *_), (hg, *_) = model(x, v_spme)
        soc_pred.append(sg.cpu().numpy())
        soc_true.append(soc.numpy())
        soh_pred.append(hg.cpu().numpy())
        soh_true.append(soh.numpy())

    sp = np.concatenate(soc_pred)
    st = np.concatenate(soc_true)
    hp = np.concatenate(soh_pred)
    ht = np.concatenate(soh_true)

    soc_rmse = float(np.sqrt(np.mean((sp - st) ** 2))) * 100.0  # fraction → %
    soh_rmse = float(np.sqrt(np.mean((hp - ht) ** 2))) * 100.0
    soc_r = float(np.corrcoef(sp, st)[0, 1]) if len(sp) > 1 else float("nan")
    soh_r = float(np.corrcoef(hp, ht)[0, 1]) if len(hp) > 1 else float("nan")
    return {"soc_rmse": soc_rmse, "soh_rmse": soh_rmse,
            "soc_r": soc_r, "soh_r": soh_r, "n": len(sp)}


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def run_fold(cell, args, device):
    print(f"\n=== Fold: held_out={cell} ===\n", flush=True)

    train_ds = NASABatteryDataset(args.data_dir, split="train",
                                  split_mode="loco", held_out_cell=cell)
    val_ds   = NASABatteryDataset(args.data_dir, split="val",
                                  split_mode="loco", held_out_cell=cell)
    test_ds  = NASABatteryDataset(args.data_dir, split="test",
                                  split_mode="loco", held_out_cell=cell)

    print(f"  train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}", flush=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin)

    model = UAPIFormer(**MODEL_KWARGS).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    ckpt_dir = os.path.join("checkpoints", f"loco_{cell}")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_soc   = float("inf")
    best_epoch     = 0
    best_state     = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()

        val_soc, val_soh = _val_soc_rmse(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_SOC={val_soc:.4f}% | "
              f"val_SOH={val_soh:.4f}%", flush=True)

        if val_soc < best_val_soc:
            best_val_soc   = val_soc
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            # Save best checkpoint immediately so it survives a crash.
            torch.save(
                {"model_state": best_state, "epoch": best_epoch,
                 "val_soc_rmse": best_val_soc, "args": MODEL_KWARGS},
                os.path.join(ckpt_dir, "best.pt"),
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).", flush=True)
                break

    print(f"  Best epoch={best_epoch}, val_SOC_RMSE={best_val_soc:.4f}%", flush=True)

    # Restore best weights before test evaluation.
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate(model, test_loader, device)
    print(f"  Test  SOC={metrics['soc_rmse']:.4f}%  SOH={metrics['soh_rmse']:.4f}%  "
          f"r_soc={metrics['soc_r']:.4f}  r_soh={metrics['soh_r']:.4f}  "
          f"n={metrics['n']}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Summary table + JSON output
# ---------------------------------------------------------------------------

def print_summary(fold_results):
    keys  = ["soc_rmse", "soh_rmse", "soc_r", "soh_r"]
    cells = list(fold_results.keys())

    means = {k: float(np.mean([fold_results[c][k] for c in cells])) for k in keys}
    stds  = {k: float(np.std( [fold_results[c][k] for c in cells], ddof=0)) for k in keys}

    header = (
        f"{'Held-Out':<10}  {'SOC RMSE%':>10}  {'SOH RMSE%':>10}  "
        f"{'SOC r':>7}  {'SOH r':>7}  {'N':>6}"
    )
    sep = "=" * 62
    row_sep = "-" * 10 + "  " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 7 + "  " + "-" * 7 + "  " + "-" * 5

    print(f"\n{sep}")
    print("  LOCO Cross-Validation Results")
    print(f"  {header}")
    print(f"  {row_sep}")
    for c in cells:
        r = fold_results[c]
        print(f"  {c:<10}  {r['soc_rmse']:>10.2f}  {r['soh_rmse']:>10.2f}  "
              f"{r['soc_r']:>7.4f}  {r['soh_r']:>7.4f}  {r['n']:>6d}")
    print(f"  {row_sep}")
    print(f"  {'Mean':<10}  {means['soc_rmse']:>10.2f}  {means['soh_rmse']:>10.2f}  "
          f"{means['soc_r']:>7.4f}  {means['soh_r']:>7.4f}")
    print(f"  {'Std':<10}  {stds['soc_rmse']:>10.2f}  {stds['soh_rmse']:>10.2f}  "
          f"{stds['soc_r']:>7.4f}  {stds['soh_r']:>7.4f}")
    print(sep, flush=True)

    return means, stds


def write_json(fold_results, means, stds):
    os.makedirs("results", exist_ok=True)
    out = {
        "folds": fold_results,
        "mean": means,
        "std":  stds,
    }
    path = os.path.join("results", "loco_cv_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results written to {path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LOCO cross-validation for UAPI-Former on NASA battery cells."
    )
    p.add_argument("--held-out", type=str, default=None, choices=CELLS,
                   help="Run only this fold (for parallel GPU execution). "
                        "Omit to run all 4 folds sequentially.")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15,
                   help="Early-stopping patience on val SOC RMSE.")
    p.add_argument("--data-dir",   type=str,   default="data/raw/nasa")
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    cells_to_run = [args.held_out] if args.held_out else CELLS

    fold_results = {}
    for cell in cells_to_run:
        fold_results[cell] = run_fold(cell, args, device)

    # Print summary and write JSON only when all 4 folds are available.
    all_done = set(fold_results.keys()) == set(CELLS)
    if all_done:
        means, stds = print_summary(fold_results)
        write_json(fold_results, means, stds)
    elif len(cells_to_run) > 1 or args.held_out is None:
        # Partial summary for debugging mid-run.
        means, stds = print_summary(fold_results)


if __name__ == "__main__":
    main()

"""Training + evaluation script for the CNN-BiLSTM baseline (uapi_former/baselines.py).

Purpose: produce a LOCALLY REPRODUCED SOC/SOH RMSE for the CNN-BiLSTM SOTA
baseline (Chen et al. 2023, IEEE TII) under the EXACT SAME NASA data
pipeline and intra_cell_random split that UAPI-Former (checkpoints/nasa_v12)
uses, so the paper's headline comparison is apples-to-apples rather than
"our split vs. their split". See uapi_former/baselines.py module docstring
for the documented architectural assumptions.

This is a NEW script. It does not import or modify train.py, model.py, or
touch checkpoints/nasa_v12 / results/ablation_v12.csv.

Data pipeline match to UAPI-Former v12-MSE (checkpoints/nasa_v12/best.pt args):
    data_dir=data/raw/nasa, dataset=nasa, split_mode=intra_cell_random,
    seq_len=200, batch_size=256, lr=1e-3 (AdamW, weight_decay=1e-4),
    CosineAnnealingLR(T_max=epochs), epochs=400.
Only the INPUT CHANNELS differ: this baseline uses channels [0:3]
(V_norm, I, T_norm) of the 6-channel NASA tensor, dropping the
physics-derived channels (V_ocv_norm, soc_from_ocv, Q_cum_norm) that are
specific to UAPI-Former's own EITE design (see baselines.py docstring,
assumption #1).

Usage — smoke test (ALWAYS run this first; ~seconds, CPU or GPU):
    py scripts/train_baseline.py --smoke-test

Usage — full run (matches UAPI-Former's training budget):
    py scripts/train_baseline.py --data-dir data/raw/nasa --dataset nasa \\
        --split-mode intra_cell_random --epochs 400 --batch-size 256 \\
        --results-out results/baseline_cnn_bilstm.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from uapi_former.dataset import SyntheticDataset, NASABatteryDataset, CALCEDataset, OxfordDataset
from uapi_former.baselines import CNNBiLSTM, count_params


# ---------------------------------------------------------------------------
# Channel-subset wrapper: reuses uapi_former/dataset.py Dataset classes
# UNCHANGED, just slices to the first N channels (V_norm, I, T_norm) so the
# CNN-BiLSTM baseline never sees UAPI-Former's physics-derived channels.
# ---------------------------------------------------------------------------
class ChannelSubsetDataset(Dataset):
    def __init__(self, base_dataset, n_channels: int = 3):
        self.base = base_dataset
        self.n_channels = n_channels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if len(item) == 4:
            x, _v_spme, soc, soh = item
        else:
            x, soc, soh = item
        x = x[:, : self.n_channels].contiguous().float()
        return x, soc, soh


def train_epoch(model, dataloader, optimizer, scaler, device,
                 w_soc: float = 10.0, w_soh: float = 2.0):
    model.train()
    total_loss, n = 0.0, 0
    for x, soc, soh in dataloader:
        x   = x.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            soc_pred, soh_pred = model(x)
            loss = w_soc * F.mse_loss(soc_pred, soc) + w_soh * F.mse_loss(soh_pred, soh)

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
def eval_monitor_loss(model, dataloader, device):
    """5*MSE_soc + MSE_soh, same convention as train.py's _eval_loss, used
    ONLY for best-checkpoint selection (not reported as a final metric)."""
    model.eval()
    total_soc_mse, total_soh_mse, n = 0.0, 0.0, 0
    for x, soc, soh in dataloader:
        x   = x.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)
        soc_pred, soh_pred = model(x)
        bs = x.size(0)
        total_soc_mse += F.mse_loss(soc_pred, soc).item() * bs
        total_soh_mse += F.mse_loss(soh_pred, soh).item() * bs
        n += bs
    model.train()
    soc_mse = total_soc_mse / max(1, n)
    soh_mse = total_soh_mse / max(1, n)
    return 5.0 * soc_mse + soh_mse


@torch.no_grad()
def evaluate_test_set(model, dataloader, device):
    """Full test-set evaluation: RMSE, MAE, bias, Pearson r for SOC and SOH."""
    model.eval()
    soc_pred, soc_true, soh_pred, soh_true = [], [], [], []
    for x, soc, soh in dataloader:
        x = x.to(device)
        sp, hp = model(x)
        soc_pred.append(sp.cpu().numpy())
        soc_true.append(soc.numpy())
        soh_pred.append(hp.cpu().numpy())
        soh_true.append(soh.numpy())
    sp = np.concatenate(soc_pred); st = np.concatenate(soc_true)
    hp = np.concatenate(soh_pred); ht = np.concatenate(soh_true)

    def stats(pred, true):
        err = pred - true
        return {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae":  float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "std":  float(np.std(err)),
            "r":    float(np.corrcoef(pred, true)[0, 1]),
            "n":    int(len(true)),
        }

    return {"soc": stats(sp, st), "soh": stats(hp, ht)}


def build_datasets(args):
    """Returns (train_ds, val_ds, test_ds) wrapped to `args.in_channels` channels."""
    if args.synthetic:
        train_ds = SyntheticDataset(num_samples=args.num_synthetic, seq_len=args.seq_len,
                                     in_channels=args.in_channels, seed=1)
        val_ds   = SyntheticDataset(num_samples=max(256, args.num_synthetic // 8), seq_len=args.seq_len,
                                     in_channels=args.in_channels, seed=2)
        test_ds  = SyntheticDataset(num_samples=max(256, args.num_synthetic // 8), seq_len=args.seq_len,
                                     in_channels=args.in_channels, seed=3)
        return train_ds, val_ds, test_ds

    ds_cls_map = {"nasa": NASABatteryDataset, "calce": CALCEDataset, "oxford": OxfordDataset}
    ds_cls = ds_cls_map[args.dataset]
    common = dict(seq_len=args.seq_len, spme_cache=None)
    if args.dataset == "nasa":
        common["split_mode"] = args.split_mode

    train_raw = ds_cls(args.data_dir, split="train", **common)
    val_raw   = ds_cls(args.data_dir, split="val",   **common)
    test_raw  = ds_cls(args.data_dir, split="test",  **common)

    train_ds = ChannelSubsetDataset(train_raw, n_channels=args.in_channels)
    val_ds   = ChannelSubsetDataset(val_raw,   n_channels=args.in_channels)
    test_ds  = ChannelSubsetDataset(test_raw,  n_channels=args.in_channels)
    return train_ds, val_ds, test_ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    type=str, default="data/raw/nasa")
    p.add_argument("--dataset",     type=str, default="nasa", choices=["nasa", "calce", "oxford"])
    p.add_argument("--split-mode",  type=str, default="intra_cell_random",
                   choices=["cross_cell", "intra_cell", "intra_cell_random"])
    p.add_argument("--seq-len",     type=int, default=200)
    p.add_argument("--in-channels", type=int, default=3,
                   help="3 = [V_norm, I, T_norm] raw channels only (no physics-derived features).")
    p.add_argument("--epochs",      type=int, default=400)
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/baseline_cnn_bilstm")
    p.add_argument("--results-out", type=str, default="results/baseline_cnn_bilstm.json")
    p.add_argument("--synthetic",   action="store_true", help="Force synthetic dataset.")
    p.add_argument("--num-synthetic", type=int, default=4096)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--smoke-test",  action="store_true",
                   help="Convenience flag: synthetic data, 2 epochs, writes to a "
                        "_smoke_test suffixed checkpoint dir / results file so the "
                        "smoke-test numbers can NEVER be mistaken for / overwrite "
                        "the real reported results (Rule: never use smoke-test "
                        "results in the manuscript).")
    args = p.parse_args()

    if args.smoke_test:
        args.synthetic = True
        args.epochs = 2
        args.num_synthetic = 512
        args.checkpoint_dir = args.checkpoint_dir.rstrip("/\\") + "_smoke_test"
        base, ext = os.path.splitext(args.results_out)
        args.results_out = f"{base}_smoke_test{ext}"
        print("=" * 70)
        print("  SMOKE TEST MODE — synthetic data, 2 epochs. NOT a reportable result.")
        print(f"  Checkpoint dir : {args.checkpoint_dir}")
        print(f"  Results out    : {args.results_out}")
        print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = build_datasets(args)
    print(f"Dataset sizes — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=(device.type == "cuda"))

    model = CNNBiLSTM(in_channels=args.in_channels, seq_len=args.seq_len).to(device)
    n_params = count_params(model)
    print(f"CNN-BiLSTM parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0

    wall_clock_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        val_loss = eval_monitor_loss(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            approx_rmse = (val_loss / 6.0) ** 0.5
            print(f"Epoch {epoch:4d}/{args.epochs} | train: {train_loss:.6f} | "
                  f"val_mse: {val_loss:.6f} (~{approx_rmse*100:.2f}% SOC RMSE) | "
                  f"lr: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss, "val_loss": val_loss,
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, "best.pt"))

    train_wall_clock_s = time.time() - wall_clock_start
    torch.save({"epoch": args.epochs, "model_state": model.state_dict(), "args": vars(args)},
               os.path.join(args.checkpoint_dir, "last.pt"))
    print(f"Training complete in {train_wall_clock_s:.1f}s. "
          f"Best val monitor loss: {best_loss:.6f} @ epoch {best_epoch}.", flush=True)

    # ---- Final test-set evaluation using the BEST checkpoint (by val loss) ----
    best_ckpt = torch.load(os.path.join(args.checkpoint_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_stats = evaluate_test_set(model, test_loader, device)

    print("=" * 70)
    print(f"  TEST SET RESULTS (best checkpoint, epoch {best_ckpt['epoch']})")
    print(f"  SOC RMSE = {test_stats['soc']['rmse']*100:.2f}%  MAE = {test_stats['soc']['mae']*100:.2f}%  "
          f"r = {test_stats['soc']['r']:.4f}  bias = {test_stats['soc']['bias']*100:+.3f}%")
    print(f"  SOH RMSE = {test_stats['soh']['rmse']:.4f}  MAE = {test_stats['soh']['mae']:.4f}  "
          f"r = {test_stats['soh']['r']:.4f}  bias = {test_stats['soh']['bias']:+.4f}")
    print("=" * 70)

    results = {
        "model": "CNN-BiLSTM (reproduced, this work)",
        "architecture_note": "Standard-choices reconstruction — see uapi_former/baselines.py "
                              "docstring for documented assumptions; NOT the exact architecture "
                              "of the cited paper (hyperparameters not fully specified there).",
        "literature_citation": "Chen, Zhao, Zhang, Shu, Shen, Liu — IEEE Trans. Ind. Informatics, "
                                "vol 19, pp 8352-8362 (2023). Literature-reported SOC RMSE: 0.49% "
                                "(their own train/test split, NOT this split).",
        "dataset": args.dataset if not args.synthetic else "synthetic",
        "data_dir": args.data_dir,
        "split_mode": args.split_mode,
        "seq_len": args.seq_len,
        "in_channels": args.in_channels,
        "in_channels_desc": "[V_norm, I, T_norm] — raw physical channels only, no SPMe/physics-derived features",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "param_count": n_params,
        "epochs_requested": args.epochs,
        "best_epoch": best_ckpt["epoch"],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "AdamW(weight_decay=1e-4) + CosineAnnealingLR",
        "loss": "10*MSE(SOC) + 2*MSE(SOH) (same weighting as UAPI-Former v12-MSE canonical run)",
        "device": str(device),
        "train_wall_clock_seconds": train_wall_clock_s,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "checkpoint_path": os.path.join(args.checkpoint_dir, "best.pt"),
        "test_soc_rmse": test_stats["soc"]["rmse"],
        "test_soc_rmse_pct": test_stats["soc"]["rmse"] * 100.0,
        "test_soc_mae": test_stats["soc"]["mae"],
        "test_soc_bias": test_stats["soc"]["bias"],
        "test_soc_r": test_stats["soc"]["r"],
        "test_soh_rmse": test_stats["soh"]["rmse"],
        "test_soh_mae": test_stats["soh"]["mae"],
        "test_soh_bias": test_stats["soh"]["bias"],
        "test_soh_r": test_stats["soh"]["r"],
        "full_stats": test_stats,
    }

    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.results_out}")


if __name__ == "__main__":
    main()

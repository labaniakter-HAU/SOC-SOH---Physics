"""Training script for UAPI-Former.

Supports both the synthetic dataset (quick smoke-test) and the real memmap
dataset produced by scripts/make_memmap.py.

Canonical Paper B configuration (v12+):
    d_model=128, nhead=4, num_layers=4, seq_len=200, in_channels=6
    6 channels: [V_norm, I, T_norm, V_ocv_norm, soc_from_ocv, Q_cum_norm]

Example — synthetic smoke-test (CPU):
    py train.py --epochs 1 --batch-size 64 --device cpu --synthetic

Example — real data (GPU, fp16):
    py train.py --data-dir data/memmap/nasa --epochs 200 --batch-size 256
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uapi_former.dataset import (SyntheticDataset, MemmapDataset,
                                 NASABatteryDataset, CALCEDataset, OxfordDataset,
                                 MITTRIDataset)
from uapi_former.model import UAPIFormer
from uapi_former.evidential import nig_loss


def train_epoch(model, dataloader, optimizer, scaler, device,
                nig_coeff: float = 0.1, mse_only: bool = False):
    model.train()
    total_loss, n = 0.0, 0
    for batch in dataloader:
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
            if mse_only:
                loss = 10.0 * F.mse_loss(sg, soc) + 2.0 * F.mse_loss(hg, soh)
            else:
                # NIG evidential loss + MSE anchor to prevent lazy-mean collapse
                loss = (nig_loss(soc, sg, sv, sa, sb, coeff=nig_coeff)
                        + nig_loss(soh, hg, hv, ha, hb, coeff=nig_coeff)
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
def _eval_loss(model, dataloader, device):
    """Returns negative mean SOC MSE for checkpoint selection.

    Using the MSE of the predicted mean (γ) rather than the full NIG ELBO avoids
    the NIG collapse where ν→0 gives very negative ELBO but the mean degrades.
    Lower return value = better SOC accuracy.
    """
    model.eval()
    total_soc_mse, total_soh_mse, n = 0.0, 0.0, 0
    for batch in dataloader:
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
        total_soc_mse += F.mse_loss(sg, soc).item() * bs
        total_soh_mse += F.mse_loss(hg, soh).item() * bs
        n += bs
    model.train()
    # Combined MSE: weight SOC 5× more than SOH (primary target).
    # Return negative so that lower RMSE = lower (better) monitor_loss.
    soc_mse = total_soc_mse / max(1, n)
    soh_mse = total_soh_mse / max(1, n)
    return 5.0 * soc_mse + soh_mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=str,   default=None,
                   help="Path to raw data dir (nasa/calce/oxford) or memmap folder.")
    p.add_argument("--dataset",    type=str,   default=None,
                   choices=["nasa", "calce", "oxford", "mit_tri", "all", "memmap"],
                   help="Dataset type when --data-dir is a raw directory. "
                        "'all' combines nasa+calce+oxford from --data-dir parent.")
    p.add_argument("--spme-cache", type=str,   default=None,
                   help="Path to precomputed SPMe .npz cache.")
    p.add_argument("--synthetic",  action="store_true",
                   help="Force synthetic dataset even if --data-dir is set.")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len",    type=int,   default=200)
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--nhead",      type=int,   default=4)
    p.add_argument("--num-layers", type=int,   default=4)
    p.add_argument("--in-channels",type=int,   default=6,
                   help="6 = full (V_norm,I,T_norm,V_ocv_norm,soc_from_ocv,Q_cum_norm); "
                        "5 = legacy SPMe-augmented; 3 = no-EITE ablation.")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--num-synthetic",  type=int, default=4096,
                   help="Number of synthetic samples when --synthetic is used.")
    p.add_argument("--split-mode",    type=str, default="cross_cell",
                   choices=["cross_cell", "intra_cell", "intra_cell_random"],
                   help="cross_cell = train B0005+B0006/val B0007/test B0018; "
                        "intra_cell = all 4 cells split by cycle fraction 70/15/15; "
                        "intra_cell_random = all 4 cells with random 70/15/15 cycle assignment (SOTA-comparable).")
    p.add_argument("--nig-coeff",     type=float, default=0.1,
                   help="NIG evidence regularization coefficient (coeff in nig_loss). "
                        "Lower values (e.g. 0.01) reduce NIG oscillation while still training "
                        "uncertainty parameters. MSE anchor dominates when coeff << 1.")
    p.add_argument("--mse-only",      action="store_true",
                   help="Train with pure MSE loss (no NIG ELBO). Eliminates NIG oscillation. "
                        "Use conformal calibration post-hoc for uncertainty quantification.")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to a checkpoint (best.pt) to resume training from. "
                        "Restores model, optimizer, scheduler, and best_loss so training "
                        "continues from the saved epoch with the same LR schedule.")
    p.add_argument("--ft",            action="store_true",
                   help="Fine-tune mode: when used with --resume, resets best_loss, "
                        "optimizer, and LR schedule so the model adapts to a new dataset "
                        "from epoch 1 with the current --lr.")
    p.add_argument("--seed",          type=int,   default=None,
                   help="Random seed for reproducibility (torch/cuda/numpy/random). "
                        "When set, seeds are applied before any model/data initialization.")
    p.add_argument("--ft-max-cycles", type=int,   default=None,
                   help="Few-shot fine-tuning: cap the CALCE training set to the first "
                        "N labelled cycles per cell (None = use full training set).")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    val_dataset = None  # set per-branch below
    if args.synthetic or args.data_dir is None:
        dataset = SyntheticDataset(num_samples=args.num_synthetic, seq_len=args.seq_len,
                                   in_channels=args.in_channels)
        print(f"Using synthetic dataset ({len(dataset)} samples, {args.in_channels} channels).")
    elif args.dataset == "nasa" or (args.dataset is None and "nasa" in args.data_dir.lower()):
        sm = args.split_mode
        dataset = NASABatteryDataset(args.data_dir, split="train",
                                     seq_len=args.seq_len, spme_cache=args.spme_cache,
                                     split_mode=sm)
        val_dataset = NASABatteryDataset(args.data_dir, split="val",
                                         seq_len=args.seq_len, spme_cache=args.spme_cache,
                                         split_mode=sm)
        print(f"NASA dataset ({sm}): {len(dataset)} train, {len(val_dataset)} val samples.", flush=True)
    elif args.dataset == "calce" or (args.dataset is None and "calce" in args.data_dir.lower()):
        dataset = CALCEDataset(args.data_dir, split="train",
                               seq_len=args.seq_len, spme_cache=args.spme_cache,
                               max_cycles_per_cell=args.ft_max_cycles)
        val_dataset = CALCEDataset(args.data_dir, split="val",
                                   seq_len=args.seq_len, spme_cache=args.spme_cache)
        cyc_note = f" ({args.ft_max_cycles} cycles/cell few-shot cap)" if args.ft_max_cycles else ""
        print(f"CALCE dataset{cyc_note}: {len(dataset)} train, {len(val_dataset)} val samples.")
    elif args.dataset == "oxford" or (args.dataset is None and "oxford" in args.data_dir.lower()):
        dataset = OxfordDataset(args.data_dir, split="train",
                                seq_len=args.seq_len, spme_cache=args.spme_cache)
        val_dataset = OxfordDataset(args.data_dir, split="val",
                                    seq_len=args.seq_len, spme_cache=args.spme_cache)
        print(f"Oxford dataset: {len(dataset)} train, {len(val_dataset)} val samples.")
    elif args.dataset == "mit_tri" or (args.dataset is None and "mit_tri" in args.data_dir.lower()):
        dataset = MITTRIDataset(args.data_dir, split="train",
                                seq_len=args.seq_len, spme_cache=args.spme_cache)
        val_dataset = MITTRIDataset(args.data_dir, split="val",
                                    seq_len=args.seq_len, spme_cache=args.spme_cache)
        print(f"MIT-TRI dataset: {len(dataset)} train, {len(val_dataset)} val samples.")
    elif args.dataset == "all":
        from torch.utils.data import ConcatDataset
        base = args.data_dir
        datasets_train, datasets_val = [], []
        for ds_cls, sub in [(NASABatteryDataset, "nasa"), (CALCEDataset, "calce"), (OxfordDataset, "oxford")]:
            p_sub = os.path.join(base, sub)
            if os.path.isdir(p_sub):
                try:
                    datasets_train.append(ds_cls(p_sub, "train", args.seq_len, args.spme_cache))
                    datasets_val.append(ds_cls(p_sub, "val",   args.seq_len, args.spme_cache))
                except Exception as e:
                    print(f"Warning: skipping {sub}: {e}")
        dataset     = ConcatDataset(datasets_train)
        val_dataset = ConcatDataset(datasets_val)
        print(f"Combined dataset: {len(dataset)} train, {len(val_dataset)} val samples.")
    else:
        dataset     = MemmapDataset(args.data_dir)
        val_dataset = None
        print(f"Memmap dataset from {args.data_dir} ({len(dataset)} samples.)")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = UAPIFormer(
        in_channels=args.in_channels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss  = float("inf")
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        ckpt_r = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt_r["model_state"])
        if args.ft:
            # Fine-tune: keep only model weights, reset everything else
            print(f"Fine-tuning from {args.resume} — resetting optimizer/scheduler/best_loss", flush=True)
        else:
            if "optimizer_state" in ckpt_r:
                optimizer.load_state_dict(ckpt_r["optimizer_state"])
            start_epoch = ckpt_r.get("epoch", 0) + 1
            best_loss   = ckpt_r.get("val_loss", float("inf"))
            # Advance the cosine scheduler to the correct position without stepping optimizer
            for _ in range(start_epoch - 1):
                scheduler.step()
            print(f"Resumed from {args.resume} (epoch {start_epoch-1}, best_loss={best_loss:.6f})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, scaler, device,
                                 nig_coeff=args.nig_coeff, mse_only=args.mse_only)
        scheduler.step()

        # Validation loss (used for checkpoint selection when available)
        if val_loader is not None:
            val_loss    = _eval_loss(model, val_loader, device)
            monitor_loss = val_loss
        else:
            val_loss     = None
            monitor_loss = train_loss

        if epoch % 10 == 0 or epoch == 1:
            if val_loss is not None:
                # val_loss = 5*MSE_soc + MSE_soh; approx SOC RMSE = sqrt(val_loss/6)
                approx_rmse = (val_loss / 6.0) ** 0.5
                print(f"Epoch {epoch:4d}/{args.epochs} | train: {train_loss:.6f} | "
                      f"val_mse: {val_loss:.6f} (~{approx_rmse*100:.1f}% SOC RMSE) | "
                      f"lr: {scheduler.get_last_lr()[0]:.2e}", flush=True)
            else:
                print(f"Epoch {epoch:4d}/{args.epochs} | loss: {train_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))

    torch.save({"epoch": args.epochs, "model_state": model.state_dict(), "args": vars(args)},
               os.path.join(args.checkpoint_dir, "last.pt"))
    print(f"Training complete. Best loss: {best_loss:.6f}. Checkpoints in {args.checkpoint_dir}/", flush=True)


if __name__ == "__main__":
    main()

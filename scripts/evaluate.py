"""Evaluate a trained UAPI-Former checkpoint on a real or synthetic dataset.

Computes per-task RMSE, MAE, and ECE (Expected Calibration Error) at 95% CI.
Optionally saves a reliability diagram (calibration plot) to disk.

Usage:
    py scripts\\evaluate.py --checkpoint checkpoints/best.pt --split test
    py scripts\\evaluate.py --checkpoint checkpoints/best.pt --dataset nasa \\
        --data-dir data/raw/nasa --split test --spme-cache cache/spme_nmc.npz \\
        --plot calibration.png
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from uapi_former.model import UAPIFormer
from uapi_former.eval import evaluate_model
from uapi_former.metrics import rmse, mae, ece_from_nig
from uapi_former.evidential import nig_predictive_variance


def load_dataset(args):
    if args.dataset == "synthetic":
        from uapi_former.dataset import SyntheticDataset
        return SyntheticDataset(num_samples=args.num_synthetic, seq_len=args.seq_len,
                                in_channels=args.in_channels)
    elif args.dataset == "nasa":
        from uapi_former.dataset import NASABatteryDataset
        extra = {"held_out_cell": args.held_out_cell} if args.split_mode == "loco" else {}
        return NASABatteryDataset(args.data_dir, split=args.split, seq_len=args.seq_len,
                                  spme_cache=args.spme_cache, split_mode=args.split_mode,
                                  **extra)
    elif args.dataset == "calce":
        from uapi_former.dataset import CALCEDataset
        return CALCEDataset(args.data_dir, split=args.split, seq_len=args.seq_len,
                            spme_cache=args.spme_cache)
    elif args.dataset == "mit_tri":
        from uapi_former.dataset import MITTRIDataset
        return MITTRIDataset(args.data_dir, split=args.split, seq_len=args.seq_len,
                             spme_cache=args.spme_cache)
    elif args.dataset == "oxford":
        from uapi_former.dataset import OxfordDataset
        return OxfordDataset(args.data_dir, split=args.split, seq_len=args.seq_len,
                             spme_cache=args.spme_cache)
    elif args.dataset == "memmap":
        from uapi_former.dataset import MemmapDataset
        return MemmapDataset(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def save_reliability_diagram(gamma, var, y, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        probs = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        from uapi_former.metrics import ece_from_mean_var
        _, coverages = ece_from_mean_var(gamma, var, y, probs=probs)

        nominal  = list(coverages.keys())
        observed = list(coverages.values())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(nominal, observed, "o-", label="Model")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_title("Calibration reliability diagram")
        ax.legend()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Reliability diagram saved to {out_path}")
    except Exception as e:
        print(f"Warning: could not save reliability diagram: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--dataset",       type=str, default="synthetic",
                   choices=["synthetic", "nasa", "calce", "mit_tri", "oxford", "memmap"])
    p.add_argument("--data-dir",      type=str, default=None)
    p.add_argument("--split",         type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--spme-cache",    type=str, default=None)
    p.add_argument("--device",        type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size",    type=int, default=256)
    p.add_argument("--seq-len",       type=int, default=200)
    p.add_argument("--in-channels",   type=int, default=6)
    p.add_argument("--split-mode",    type=str, default="intra_cell_random",
                   choices=["cross_cell", "intra_cell", "intra_cell_random", "loco"])
    p.add_argument("--held-out-cell", type=str, default="B0005",
                   help="Held-out cell for split-mode=loco (must match the fold "
                        "the checkpoint was trained on).")
    p.add_argument("--d-model",       type=int, default=128)
    p.add_argument("--nhead",         type=int, default=4)
    p.add_argument("--num-layers",    type=int, default=4)
    p.add_argument("--num-synthetic", type=int, default=1024)
    p.add_argument("--plot",          type=str, default=None,
                   help="Path to save calibration reliability diagram (PNG).")
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})
    in_ch  = saved_args.get("in_channels",  args.in_channels)
    dm     = saved_args.get("d_model",      args.d_model)
    nh     = saved_args.get("nhead",        args.nhead)
    nl     = saved_args.get("num_layers",   args.num_layers)
    sl     = saved_args.get("seq_len",      args.seq_len)

    model = UAPIFormer(in_channels=in_ch, d_model=dm, nhead=nh, num_layers=nl, seq_len=sl)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device)

    dataset = load_dataset(args)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = evaluate_model(model, loader, device)

    print(f"\n{'='*55}")
    print(f"  Dataset : {args.dataset}  |  split : {args.split}")
    print(f"  Samples : {len(dataset)}")
    print(f"{'='*55}")
    print(f"  SOC  RMSE : {results['soc_rmse']:.4f}   MAE : {results.get('soc_mae', float('nan')):.4f}")
    print(f"  SOH  RMSE : {results['soh_rmse']:.4f}   MAE : {results.get('soh_mae', float('nan')):.4f}")
    print(f"  SOC  ECE  : {results['soc_ece']:.4f}")
    print(f"  SOH  ECE  : {results['soh_ece']:.4f}")
    print(f"{'='*55}\n")

    # Conformal coverage reporting (only when calibrated checkpoint used)
    calib = ckpt.get("calibration", {})
    if "conformal_q_soc_90" in calib and "soc_gamma" in results:
        soc_sigma = torch.sqrt(results["soc_var"].clamp(min=1e-8))
        soh_sigma = torch.sqrt(results["soh_var"].clamp(min=1e-8))
        print(f"  {'Conformal coverage':-<45}")
        for pct in (90, 95):
            q_soc = calib[f"conformal_q_soc_{pct}"]
            q_soh = calib[f"conformal_q_soh_{pct}"]
            soc_cov = (torch.abs(results["soc_true"] - results["soc_gamma"]) <= q_soc * soc_sigma).float().mean()
            soh_cov = (torch.abs(results["soh_true"] - results["soh_gamma"]) <= q_soh * soh_sigma).float().mean()
            print(f"  [{pct}%]  SOC: {float(soc_cov):.3f} (nominal {pct/100:.2f})  "
                  f"SOH: {float(soh_cov):.3f} (nominal {pct/100:.2f})")
        print(f"{'='*55}\n")

    if args.plot and "soh_gamma" in results:
        save_reliability_diagram(
            results["soh_gamma"], results["soh_var"], results["soh_true"], args.plot
        )

    return results


if __name__ == "__main__":
    main()

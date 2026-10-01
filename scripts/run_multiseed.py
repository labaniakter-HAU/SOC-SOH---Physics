"""Multi-seed reproducibility runner for UAPI-Former (Q1 robustness check).

Trains and evaluates ONE of three model configs — the "Full" model (matching
the Full row of results/ablation_v12.csv) or one of the two largest-delta
ablation variants (No-DTAG+CTBA, No-EITE) — across a list of seeds, using the
EXACT SAME training/eval code path as scripts/ablation.py's run_variant():
200 epochs (by default), AdamW (lr=1e-3, wd=1e-4), CosineAnnealingLR,
MSE-only loss, NASA intra_cell_random split, no validation-based early
stopping (final-epoch weights are evaluated once on the held-out test set —
identical protocol to the original single-seed ablation_v12.csv numbers).

This script imports scripts/ablation.py's model classes and its
_train_one_epoch / _evaluate helpers directly rather than duplicating them,
so the training code is byte-for-byte identical to what produced
results/ablation_v12.csv. It does NOT modify ablation.py or train.py, and it
NEVER writes to checkpoints/nasa_v12/ or results/ablation_v12.csv.

Only the "full" variant supports --ckpt-dir-template (per-seed checkpoint
saving requested for the Full-model multi-seed run); the ablation variants
(No-DTAG+CTBA, No-EITE) only need aggregated RMSE numbers, not checkpoints.

Usage:
    py scripts/run_multiseed.py --variant full --seeds 0 1 2 3 4 \\
        --epochs 200 --out-json results/_multiseed_raw/full.json \\
        --ckpt-dir-template "checkpoints/nasa_v12_seed{seed}"

    py scripts/run_multiseed.py --variant no_dtag_ctba --seeds 0 1 2 \\
        --epochs 200 --out-json results/_multiseed_raw/no_dtag_ctba.json

    py scripts/run_multiseed.py --variant no_eite --seeds 0 1 2 \\
        --epochs 200 --out-json results/_multiseed_raw/no_eite.json

Smoke test (2 epochs on real NASA data, confirms the pipeline works
end-to-end and that the same seed reproduces the same result):
    py scripts/run_multiseed.py --variant full --seeds 0 0 --epochs 2 \\
        --out-json <scratch>/smoke.json
"""
import argparse
import json
import os
import random
import sys
import time

# Must be set before the CUDA context is created (i.e. before the first .cuda()
# call) for torch.use_deterministic_algorithms(True) to make cuBLAS ops
# reproducible. Harmless on CPU-only runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # scripts/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import ablation as abl                       # scripts/ablation.py (adds seed flag, model classes)
from uapi_former.dataset import NASABatteryDataset
from uapi_former.model import UAPIFormer


def _full_model(sl):
    return UAPIFormer(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_dtag_ctba_model(sl):
    return abl.UAPIFormerNoDTAG(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_eite_model(sl):
    return abl.UAPIFormerNoEITE(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_nig_model(sl):
    return abl.UAPIFormerNoNIG(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_prap_model(sl):
    return abl.UAPIFormerNoPRAP(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_dct_model(sl):
    return abl.UAPIFormerNoDCT(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_ctba_model(sl):
    return abl.UAPIFormerNoCTBA(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _no_ic_model(sl):
    return abl.UAPIFormerNoIC(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


def _ctba_self_model(sl):
    return abl.UAPIFormerCTBASelf(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl)


# name -> (display name matching ablation_v12.csv, model factory, is_nig)
VARIANTS = {
    "full":         ("Full (EITE+NIG+DTAG+PRAP+DCT+CTBA+IC)", _full_model, True),
    "no_dtag_ctba": ("No-DTAG+CTBA (shared rep, no cross-task attn)", _no_dtag_ctba_model, True),
    "no_eite":      ("No-EITE (SPMe channels zeroed)", _no_eite_model, True),
    "no_nig":       ("No-NIG (MSE head)", _no_nig_model, False),
    "no_prap":      ("No-PRAP (uniform mean pooling)", _no_prap_model, True),
    "no_dct":       ("No-DCT (no chemistry token)", _no_dct_model, True),
    "no_ctba":      ("No-CTBA (no cross-task attention)", _no_ctba_model, True),
    "no_ic":        ("No-IC (no incremental capacity branch)", _no_ic_model, True),
    "ctba_self":    ("CTBA self-projection (parameter-matched, no cross-task flow)", _ctba_self_model, True),
}


def set_seed(seed: int):
    """Seed random/numpy/torch/cuda. Must be called before any model/data init.

    Also enables deterministic cuDNN/CUBLAS algorithms so that repeated runs
    with the *same* seed reproduce the same result on GPU (best-effort: some
    ops fall back with a warning rather than erroring, via warn_only=True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"[WARN] could not enable fully deterministic algorithms: {e}")
    # Flash / memory-efficient attention backward kernels are non-deterministic
    # by design (parallel-reduction atomics). Force the plain "math" SDPA
    # backend so that repeated runs with the same seed give the same result.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def run_one(variant, seed, epochs, batch_size, lr, seq_len, data_dir, device,
            ckpt_path=None):
    name, model_fn, is_nig = VARIANTS[variant]

    # Seed BEFORE any data/model construction (Rule 5 / fair-comparison discipline).
    set_seed(seed)

    train_ds = NASABatteryDataset(data_dir, split="train", seq_len=seq_len,
                                  spme_cache=None, split_mode="intra_cell_random")
    test_ds  = NASABatteryDataset(data_dir, split="test", seq_len=seq_len,
                                  spme_cache=None, split_mode="intra_cell_random")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_fn(seq_len).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sc    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    t0 = time.time()
    print(f"\n{'='*60}\n  [{variant}] seed={seed}  epochs={epochs}  "
          f"train_n={len(train_ds)}  test_n={len(test_ds)}\n{'='*60}", flush=True)
    for ep in range(1, epochs + 1):
        loss = abl._train_one_epoch(model, train_loader, opt, sc, device,
                                    is_nig=is_nig, mse_only=True)
        sch.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  [{variant} seed={seed}] epoch {ep}/{epochs}  loss={loss:.6f}", flush=True)
    elapsed = time.time() - t0

    soc_rmse, soc_mae, soh_rmse, soh_mae = abl._evaluate(model, test_loader, device, is_nig=is_nig)
    print(f"  [{variant} seed={seed}] DONE in {elapsed:.1f}s | "
          f"SOC RMSE={soc_rmse*100:.3f}%  SOH RMSE={soh_rmse*100:.3f}%", flush=True)

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "epoch": epochs,
            "model_state": model.state_dict(),
            "seed": seed,
            "variant": variant,
            "soc_rmse": soc_rmse, "soc_mae": soc_mae,
            "soh_rmse": soh_rmse, "soh_mae": soh_mae,
            "args": {
                "in_channels": 6, "d_model": 128, "nhead": 4, "num_layers": 4,
                "seq_len": seq_len, "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "mse_only": True, "dataset": "nasa", "data_dir": data_dir,
                "split_mode": "intra_cell_random", "seed": seed,
            },
            "note": ("Final-epoch weights (no validation-based checkpoint selection), "
                     "matching scripts/ablation.py run_variant() protocol exactly so "
                     "SOC/SOH RMSE is directly comparable to results/ablation_v12.csv "
                     "'Full' row. Produced by scripts/run_multiseed.py, NOT by train.py."),
        }, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}", flush=True)

    return {"variant": variant, "name": name, "seed": seed,
            "soc_rmse": soc_rmse, "soc_mae": soc_mae,
            "soh_rmse": soh_rmse, "soh_mae": soh_mae,
            "elapsed_sec": elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--data-dir", type=str, default="data/raw/nasa")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--ckpt-dir-template", type=str, default=None,
                   help="e.g. 'checkpoints/nasa_v12_seed{seed}' (only meaningful for --variant full).")
    args = p.parse_args()

    device = torch.device(args.device)
    results = []
    for seed in args.seeds:
        ckpt_path = None
        if args.ckpt_dir_template:
            ckpt_dir = args.ckpt_dir_template.format(seed=seed)
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
        r = run_one(args.variant, seed, args.epochs, args.batch_size, args.lr,
                   args.seq_len, args.data_dir, device, ckpt_path=ckpt_path)
        results.append(r)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} run(s) to {args.out_json}")


if __name__ == "__main__":
    main()

"""Quick NASA B0018 evaluation: RMSE, bias, std, correlation.

Usage:
    python scripts/eval_nasa.py --checkpoint checkpoints/nasa_v7/best.pt
    python scripts/eval_nasa.py --checkpoint checkpoints/nasa_v7/best.pt --split val
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from uapi_former.model import UAPIFormer
from uapi_former.dataset import NASABatteryDataset


@torch.no_grad()
def run_eval(ckpt_path, split, split_mode, device_str):
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("args", {})

    model = UAPIFormer(
        in_channels=saved.get("in_channels", 6),
        d_model    =saved.get("d_model",     128),
        nhead      =saved.get("nhead",       4),
        num_layers =saved.get("num_layers",  4),
        seq_len    =saved.get("seq_len",     200),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = NASABatteryDataset("data/raw/nasa", split=split, split_mode=split_mode)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    soc_pred, soc_true = [], []
    soh_pred, soh_true = [], []

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
        soc_pred.append(sg.cpu().numpy())
        soc_true.append(soc.cpu().numpy())
        soh_pred.append(hg.cpu().numpy())
        soh_true.append(soh.cpu().numpy())

    sp = np.concatenate(soc_pred)
    st = np.concatenate(soc_true)
    hp = np.concatenate(soh_pred)
    ht = np.concatenate(soh_true)

    def stats(pred, true, name):
        err   = pred - true
        rmse  = float(np.sqrt(np.mean(err**2)))
        bias  = float(np.mean(err))
        std   = float(np.std(err))
        r     = float(np.corrcoef(pred, true)[0, 1])
        print(f"  {name:4s} | RMSE={rmse:.4f} ({rmse*100:.2f}%) | "
              f"bias={bias:+.4f} | std={std:.4f} | r={r:.4f}")
        print(f"       pred [{pred.min():.3f},{pred.max():.3f}] mean={pred.mean():.3f}  "
              f"true [{true.min():.3f},{true.max():.3f}] mean={true.mean():.3f}")

    print(f"\n{'='*65}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Split      : {split}  (n={len(ds)})")
    print(f"  Epoch      : {ckpt.get('epoch', '?')}")
    print(f"{'='*65}")
    stats(sp, st, "SOC")
    stats(hp, ht, "SOH")
    print(f"{'='*65}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--split",       type=str, default="test", choices=["train","val","test"])
    p.add_argument("--split-mode",  type=str, default="cross_cell",
                   choices=["cross_cell", "intra_cell", "intra_cell_random"])
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run_eval(args.checkpoint, args.split, args.split_mode, args.device)


if __name__ == "__main__":
    main()

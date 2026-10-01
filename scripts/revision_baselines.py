"""Revision R1.5 / R1.6: baselines on equal terms, and under shift.

R1.5  Both reproduced baseline families are trained for the SAME seeds as
      UAPI-Former on the random split, instead of a single seed.
R1.6  The same baselines are trained on the clean LOCO folds, so the conformal
      analysis can be repeated with a different estimator (Section
      "Baselines under shift").

Training code, optimizer, epochs and data pipeline are shared with
scripts/train_baseline*.py; only the split and the seed vary. Baselines receive
their documented channel subsets (CNN-BiLSTM: V/I/T; PI-Transformer: + IR-corrected
voltage), none of which depend on the OCV table or the Coulomb normalization, so
the clean-LOCO feature changes do not affect them.

Usage:
  py scripts/revision_baselines.py --mode random --seeds 1 2 3 4
  py scripts/revision_baselines.py --mode loco --seeds 0
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uapi_former.dataset import NASABatteryDataset
from uapi_former.baselines import CNNBiLSTM, PITransformer
from uapi_former.ocv_fit import fit_nmc_ocv_table, n_discharge_cycles
from train_baseline import ChannelSubsetDataset

CELLS = ["B0005", "B0006", "B0007", "B0018"]
MODELS = {"cnn_bilstm": (CNNBiLSTM, 3), "pi_transformer": (PITransformer, 4)}


def make(name, ch, seq_len=200):
    cls, _ = MODELS[name]
    return cls(in_channels=ch, seq_len=seq_len)


def loaders_random(ch, bs):
    kw = dict(split_mode="intra_cell_random", seq_len=200)
    d = {s: ChannelSubsetDataset(NASABatteryDataset("data/raw/nasa", split=s, **kw), n_channels=ch)
         for s in ("train", "val", "test")}
    return ({s: DataLoader(v, batch_size=bs, shuffle=(s == "train")) for s, v in d.items()}, d)


def loaders_loco(cell, seed, ch, bs):
    src = [c for c in CELLS if c != cell]
    tp = {c: NASABatteryDataset.loco_random3_positions(c, n_discharge_cycles("data/raw/nasa", c), seed)["train"]
          for c in src}
    ocv = fit_nmc_ocv_table("data/raw/nasa", tp, r0=NASABatteryDataset.R0)
    kw = dict(split_mode="loco", held_out_cell=cell, ocv_pts=ocv, q_norm="rated",
              loco_split="random3", loco_seed=seed, seq_len=200)
    d = {s: ChannelSubsetDataset(NASABatteryDataset("data/raw/nasa", split=s, **kw), n_channels=ch)
         for s in ("train", "val", "cal", "test")}
    return ({s: DataLoader(v, batch_size=bs, shuffle=(s == "train")) for s, v in d.items()}, d)


def run(model, loaders, epochs, lr, dev, patience=15):
    """Train and restore the best-validation checkpoint.

    patience > 0: stop after `patience` epochs without validation improvement.
    patience <= 0: train the full budget (UAPI-Former's protocol).
    Returns (best_val_soc_rmse, best_epoch, epochs_trained)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(dev.type, enabled=(dev.type == "cuda"))
    best, best_state, bad, best_ep, ep = float("inf"), None, 0, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, soc, soh in loaders["train"]:
            x, soc, soh = x.to(dev), soc.to(dev).view(-1), soh.to(dev).view(-1)
            with torch.amp.autocast(dev.type, enabled=(dev.type == "cuda")):
                ps, ph = model(x)
                loss = 10.0 * F.mse_loss(ps.view(-1), soc) + 2.0 * F.mse_loss(ph.view(-1), soh)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sch.step()
        v = evaluate(model, loaders["val"], dev)["soc_rmse"]
        if v < best:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, best_ep, ep


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); P, T, PH, TH = [], [], [], []
    for x, soc, soh in loader:
        ps, ph = model(x.to(dev))
        P.append(ps.view(-1).cpu()); T.append(soc.view(-1))
        PH.append(ph.view(-1).cpu()); TH.append(soh.view(-1))
    p, t, ph, th = (torch.cat(z) for z in (P, T, PH, TH))
    return {"soc_rmse": float(100 * torch.sqrt(((p - t) ** 2).mean())),
            "soh_rmse": float(100 * torch.sqrt(((ph - th) ** 2).mean())),
            "soc_r": float(np.corrcoef(p.numpy(), t.numpy())[0, 1]),
            "n": int(len(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "loco"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stopping patience; 0 trains the full budget (UAPI-Former protocol)")
    ap.add_argument("--out", default=None, help="results JSON (default results/revision/baselines_<mode>.json)")
    ap.add_argument("--ckpt-tag", default="", help="suffix for LOCO checkpoint dirs so reruns do not overwrite")
    a = ap.parse_args()
    dev = torch.device(a.device)
    epochs = a.epochs or (400 if a.mode == "random" else 100)
    path = a.out or f"results/revision/baselines_{a.mode}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in a.models:
        ch = MODELS[name][1]
        for seed in a.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if a.mode == "random":
                ld, ds = loaders_random(ch, a.batch_size)
                key = f"{name}_seed{seed}"
                if key in res:
                    print("skip", key); continue
                t0 = time.time()
                model = make(name, ch).to(dev)
                _, best_ep, trained = run(model, ld, epochs, a.lr, dev, patience=a.patience)
                m = evaluate(model, ld["test"], dev)
                m.update(seed=seed, epochs=epochs, patience=a.patience, best_epoch=best_ep,
                         epochs_trained=trained, minutes=(time.time() - t0) / 60)
                res[key] = m
                # keep the model: the cross-protocol evaluation (R1.6) reuses it
                ckpt = f"checkpoints/baseline_{name}_seed{seed}_random{a.ckpt_tag}"
                os.makedirs(ckpt, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "in_channels": ch,
                            "seed": seed, "epochs": epochs, "patience": a.patience,
                            "best_epoch": best_ep}, os.path.join(ckpt, "best.pt"))
                print(key, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)
            else:
                for cell in a.cells:
                    key = f"{name}_s{seed}_{cell}"
                    if key in res:
                        print("skip", key); continue
                    t0 = time.time()
                    ld, ds = loaders_loco(cell, seed, ch, a.batch_size)
                    model = make(name, ch).to(dev)
                    _, best_ep, trained = run(model, ld, epochs, a.lr, dev, patience=a.patience)
                    m = evaluate(model, ld["test"], dev)
                    m.update(seed=seed, cell=cell, epochs=epochs, patience=a.patience,
                             best_epoch=best_ep, epochs_trained=trained,
                             minutes=(time.time() - t0) / 60)
                    res[key] = m
                    ckpt = f"checkpoints/baseline_{name}_s{seed}_{cell}{a.ckpt_tag}"
                    os.makedirs(ckpt, exist_ok=True)
                    torch.save({"model_state": model.state_dict(), "in_channels": ch,
                                "cell": cell, "seed": seed}, os.path.join(ckpt, "best.pt"))
                    print(key, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)
            os.makedirs("results/revision", exist_ok=True)
            json.dump(res, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Electrochemical Optimal Transport (EOT) cross-chemistry transfer.

Frozen-backbone weight-free adaptation. For each target dataset (CALCE, Oxford, MIT-TRI):
  1. Cache source (NASA) + target latents `rep` via the frozen backbone.
  2. Fit a frozen IC probe on source (rep -> dQ/dV histogram).
  3. Train a residual transducer T_psi for each phase-regularizer variant by
     entropic Sinkhorn OT on UNLABELED target-train latents.
  4. Select the best variant by target-VALIDATION SOC RMSE.
  5. Evaluate the selected variant ONCE on target-TEST. Report all variants'
     validation metrics for transparency.
  6. Save a t-SNE figure (source vs target, before vs after T_psi).

Usage:
    python scripts/eot_transfer.py --smoke            # tiny, fast sanity run
    python scripts/eot_transfer.py --dataset calce
    python scripts/eot_transfer.py --dataset oxford
    python scripts/eot_transfer.py --dataset mit_tri
"""
import argparse
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from uapi_former.model import UAPIFormer
from uapi_former.dataset import NASABatteryDataset, CALCEDataset, OxfordDataset, MITTRIDataset
from uapi_former.eot import (
    OTTransducer, encode_to_rep, decode_from_rep, ic_histogram,
    sinkhorn_loss, fit_ic_probe, apply_ic_probe, phase_regularizer,
    PHASE_VARIANTS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = {
    "calce":   (CALCEDataset,  os.path.join(ROOT, "data", "raw", "calce")),
    "oxford":  (OxfordDataset, os.path.join(ROOT, "data", "raw", "oxford")),
    "mit_tri": (MITTRIDataset, os.path.join(ROOT, "data", "raw", "mit_tri")),
}
# mit_tri zero-shot/fine-tuned measured 2026-07-11 via evaluate.py on
# checkpoints/nasa_v12/best.pt and checkpoints/mit_tri_ft/best.pt respectively
# (logs/_mit_tri/01_zeroshot_eval.txt, 03_finetuned_eval.txt). The fine-tuned
# 12.08% aggregate is dominated by one outlier test cell (cell 37 of 14; see
# results/mit_tri_cell37_diagnostic.json) -- excluding it gives 1.25% (13
# cells), matching the val-split 1.33%. We report the honest, un-excluded
# aggregate here, matching the convention for calce/oxford, and disclose the
# cell37 finding separately rather than silently dropping it.
# Reference values, NOT measured by this script: read from the evaluation logs of
# scripts/evaluate.py on the same test windows, so a relabelled or retrained target cannot
# leave a typed-in reference behind (the Oxford logs are the SP2 relabel runs).
ZERO_SHOT_LOG = {"calce": "logs/_leakfix_campaign/12_calce_zeroshot_eval.txt",
                 "oxford": "logs/revision/sp2/ox_zeroshot.log",
                 "mit_tri": "logs/_mit_tri/01_zeroshot_eval.txt"}
FINE_TUNED_LOG = {"calce": "logs/_leakfix_campaign/14_calce_finetuned_eval.txt",
                  "oxford": "logs/revision/sp2/ox_ft_eval.log",
                  "mit_tri": "logs/_mit_tri/03_finetuned_eval.txt"}


def logged_soc_rmse(rel_path):
    """SOC RMSE (%) from the last 'SOC  RMSE : x' line of an evaluation log; None if absent."""
    p = os.path.join(ROOT, rel_path)
    if not os.path.exists(p):
        return None
    hits = re.findall(r"SOC\s+RMSE\s*:\s*([0-9.]+)", open(p, encoding="utf-8", errors="replace").read())
    return round(100 * float(hits[-1]), 2) if hits else None


ZERO_SHOT_SOC = {k: logged_soc_rmse(v) for k, v in ZERO_SHOT_LOG.items()}
FINE_TUNED_SOC = {k: logged_soc_rmse(v) for k, v in FINE_TUNED_LOG.items()}    # upper bound


def load_backbone(ckpt_path: str, device: torch.device) -> UAPIFormer:
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get("args", {})
    model = UAPIFormer(
        in_channels=a.get("in_channels", 6), d_model=a.get("d_model", 128),
        nhead=a.get("nhead", 4), num_layers=a.get("num_layers", 4),
        seq_len=a.get("seq_len", 200),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)        # FROZEN backbone
    return model


@torch.no_grad()
def cache_latents(model, dataset, device, batch_size=256, max_batches=None):
    """Return (rep, ic_hist, soc, soh) tensors for all windows of `dataset`."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    reps, ics, socs, sohs = [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, v_spme, soc_y, soh_y = batch
        x = x.to(device).float()
        v_spme = v_spme.to(device).float() if torch.is_tensor(v_spme) else None
        if v_spme is not None and float(v_spme.abs().sum()) == 0.0:
            v_spme = None     # no SPMe cache -> uniform PRAP pooling
        rep, _ic_feat = encode_to_rep(model, x, v_spme)
        ic_h = ic_histogram(x)
        reps.append(rep.cpu()); ics.append(ic_h.cpu())
        socs.append(soc_y.float()); sohs.append(soh_y.float())
    return (torch.cat(reps), torch.cat(ics),
            torch.cat(socs).view(-1), torch.cat(sohs).view(-1))


@torch.no_grad()
def eval_rep(model, T, rep, ic_feat_dummy, soc, soh, device):
    """Decode warped latents to SOC/SOH and return metrics dict (%)."""
    rep = rep.to(device)
    # ic_feat for the SOH head: SOC (the adoption-gate metric) is produced from
    # soc_rep only and is independent of ic_feat; we pass a zero ic_feat of the
    # right width (known SOH simplification, recorded in docs/notes/eot_novelty.md).
    z = T(rep)
    d_ic = model.soc_head.fc.in_features // 4
    ic_feat = torch.zeros(z.shape[0], d_ic, device=device)
    soc_out, soh_out = decode_from_rep(model, z, ic_feat)
    ps = soc_out[0].cpu().numpy(); ph = soh_out[0].cpu().numpy()
    ts = soc.numpy(); th = soh.numpy()
    soc_rmse = float(np.sqrt(np.mean((ps - ts) ** 2))) * 100.0
    soh_rmse = float(np.sqrt(np.mean((ph - th) ** 2))) * 100.0
    soc_r = float(np.corrcoef(ps, ts)[0, 1]) if len(ps) > 1 else float("nan")
    soh_r = float(np.corrcoef(ph, th)[0, 1]) if len(ph) > 1 else float("nan")
    return {"soc_rmse": soc_rmse, "soh_rmse": soh_rmse,
            "soc_r": soc_r, "soh_r": soh_r, "n": len(ps)}


def train_transducer(rep_tgt_train, rep_src, W_probe, ic_tgt_train, variant,
                     device, iters=400, batch=256, reg=0.1, gamma=1.0, lam=0.1,
                     lr=1e-3, seed=42):
    torch.manual_seed(seed)
    d = rep_tgt_train.shape[1]
    T = OTTransducer(d_model=d).to(device)
    opt = torch.optim.Adam(T.parameters(), lr=lr)
    rep_tgt_train = rep_tgt_train.to(device)
    rep_src = rep_src.to(device)
    W_probe = W_probe.to(device)
    ic_tgt_train = ic_tgt_train.to(device)
    n_t, n_s = rep_tgt_train.shape[0], rep_src.shape[0]
    for _ in range(iters):
        it = torch.randint(0, n_t, (min(batch, n_t),), device=device)
        isrc = torch.randint(0, n_s, (min(batch, n_s),), device=device)
        zt, zs, ict = rep_tgt_train[it], rep_src[isrc], ic_tgt_train[it]
        opt.zero_grad()
        zw = T(zt)
        loss = sinkhorn_loss(zw, zs, reg=reg)
        loss = loss + gamma * phase_regularizer(apply_ic_probe(W_probe, zw), ict, variant)
        loss = loss + lam * T.delta(zt).pow(2).mean()
        loss.backward()
        opt.step()
    return T.eval()


def save_tsne(rep_src, rep_tgt, T, out_path, device, n=500, seed=0):
    try:
        from sklearn.manifold import TSNE
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(seed)
        si = rng.choice(len(rep_src), size=min(n, len(rep_src)), replace=False)
        ti = rng.choice(len(rep_tgt), size=min(n, len(rep_tgt)), replace=False)
        src = rep_src[si].numpy()
        before = rep_tgt[ti].numpy()
        with torch.no_grad():
            after = T(rep_tgt[ti].to(device)).cpu().numpy()
        X = np.concatenate([src, before, after], axis=0)
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=seed).fit_transform(X)
        a, b = len(src), len(src) + len(before)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(emb[:a, 0], emb[:a, 1], s=8, alpha=0.5, c="#2196F3", label="NASA source")
        ax.scatter(emb[a:b, 0], emb[a:b, 1], s=8, alpha=0.5, c="#F44336", label="Target (before $T_\\psi$)")
        ax.scatter(emb[b:, 0], emb[b:, 1], s=8, alpha=0.5, c="#4CAF50", label="Target (after $T_\\psi$)")
        ax.set_title("EOT latent alignment (t-SNE)")
        ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.tight_layout(); fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"t-SNE figure saved -> {out_path}")
    except Exception as exc:
        print(f"[WARN] t-SNE figure skipped: {exc}")


def run_dataset(name, model, device, args):
    DS, ddir = TARGETS[name]
    mb = 2 if args.smoke else None
    iters = 4 if args.smoke else args.iters

    print(f"\n=== {name.upper()} ===  caching latents (frozen backbone)")
    nasa = NASABatteryDataset(os.path.join(ROOT, "data", "raw", "nasa"),
                              split="train", split_mode="intra_cell_random")
    rep_src, ic_src, _, _ = cache_latents(model, nasa, device, max_batches=mb)
    rep_tr, ic_tr, _, _ = cache_latents(model, DS(ddir, split="train"), device, max_batches=mb)
    rep_va, _, soc_va, soh_va = cache_latents(model, DS(ddir, split="val"), device, max_batches=mb)
    rep_te, _, soc_te, soh_te = cache_latents(model, DS(ddir, split="test"), device, max_batches=mb)
    print(f"  src={len(rep_src)} train={len(rep_tr)} val={len(rep_va)} test={len(rep_te)}")

    W = fit_ic_probe(rep_src, ic_src)

    # Train + select on validation
    variant_rows = {}
    best = None
    for v in PHASE_VARIANTS:
        T = train_transducer(rep_tr, rep_src, W, ic_tr, v, device,
                             iters=iters, reg=args.reg, gamma=args.gamma, lam=args.lam)
        m_val = eval_rep(model, T, rep_va, None, soc_va, soh_va, device)
        variant_rows[v] = m_val
        print(f"  variant={v:5s}  VAL soc_rmse={m_val['soc_rmse']:.2f}%  soh_rmse={m_val['soh_rmse']:.2f}%")
        if best is None or m_val["soc_rmse"] < variant_rows[best[0]]["soc_rmse"]:
            best = (v, T)

    best_variant, best_T = best
    m_test = eval_rep(model, best_T, rep_te, None, soc_te, soh_te, device)
    zero_shot_str = f"{ZERO_SHOT_SOC[name]}%" if ZERO_SHOT_SOC[name] is not None else "NOT YET MEASURED"
    fine_tuned_str = f"{FINE_TUNED_SOC[name]}%" if FINE_TUNED_SOC[name] is not None else "NOT YET MEASURED"
    print(f"  SELECTED variant={best_variant}  TEST soc_rmse={m_test['soc_rmse']:.2f}% "
          f"soh_rmse={m_test['soh_rmse']:.2f}%  (zero-shot {zero_shot_str}, "
          f"fine-tuned {fine_tuned_str})")

    if args.save_transducer:
        save_path = args.save_transducer.replace("{ds}", name)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        torch.save({"state_dict": best_T.state_dict(), "variant": best_variant,
                   "dataset": name}, save_path)
        print(f"  saved transducer -> {save_path}")

    if not args.smoke:
        save_tsne(rep_src, rep_te, best_T,
                  os.path.join(ROOT, "figures", f"eot_alignment_{name}.pdf"), device)

    beats_zero_shot = (bool(m_test["soc_rmse"] < ZERO_SHOT_SOC[name])
                        if ZERO_SHOT_SOC[name] is not None else None)  # None = not yet measured
    return {
        "selected_variant": best_variant,
        "test": m_test,
        "validation_variants": variant_rows,
        "zero_shot_soc_rmse": ZERO_SHOT_SOC[name],
        "fine_tuned_soc_rmse": FINE_TUNED_SOC[name],
        "beats_zero_shot": beats_zero_shot,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/nasa_v12/best.pt")
    p.add_argument("--dataset", choices=["calce", "oxford", "mit_tri", "both"], default="both")
    p.add_argument("--out", default="results/eot_transfer.json")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--reg", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--save-transducer", default=None,
                   help="Path template for saving the selected transducer's "
                        "state_dict, e.g. checkpoints/eot_transducer_{ds}.pt")
    p.add_argument("--backbone-seed", type=int, default=None,
                   help="SP4: backbone checkpoints/nasa_v12_seed{s}; the zero-shot and fine-tuned "
                        "reference values are then read from that backbone's own evaluation logs "
                        "(scripts/revision_sp4_paths.py), never from another lineage's.")
    args = p.parse_args()
    if args.backbone_seed is not None:
        import revision_sp4_paths as sp
        assert args.checkpoint == sp.backbone(args.backbone_seed), (args.checkpoint, args.backbone_seed)
        ZERO_SHOT_SOC.clear(); FINE_TUNED_SOC.clear()
        ZERO_SHOT_SOC.update({k: logged_soc_rmse(sp.zeroshot_log(k, args.backbone_seed)) for k in sp.DATASETS})
        FINE_TUNED_SOC.update({k: logged_soc_rmse(sp.ft_log(k, args.backbone_seed)) for k in sp.DATASETS})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_backbone(os.path.join(ROOT, args.checkpoint), device)
    print(f"Loaded frozen backbone: {args.checkpoint}")

    # NOTE: "both" intentionally still means calce+oxford only (unchanged behavior).
    # It does NOT expand to TARGETS.keys(), so mit_tri is excluded here even though
    # it's a valid TARGETS entry; run it explicitly via --dataset mit_tri.
    names = ["calce", "oxford"] if args.dataset == "both" else [args.dataset]
    results = {n: run_dataset(n, model, device, args) for n in names}

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Merge into any existing results for OTHER datasets rather than clobbering
    # them -- a single-dataset run (e.g. --dataset mit_tri) must not silently
    # erase previously-computed calce/oxford entries in the same file.
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {args.out}")


if __name__ == "__main__":
    main()

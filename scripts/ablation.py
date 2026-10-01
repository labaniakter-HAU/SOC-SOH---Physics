"""Ablation study runner for UAPI-Former.

Tests seven independently-ablatable components per Paper B §5:
    1. EITE  — zero out SPMe channels so model sees no physics information
    2. NIG   — replace NIG heads with standard MSE regression heads
    3. DTAG+CTBA — remove task gating and cross-task attention jointly
    4. PRAP  — replace physics-residual pooling with uniform mean pooling
    5. DCT   — remove chemistry fingerprint [CLS] token
    6. CTBA  — remove cross-task bidirectional attention (DTAG intact)
    7. IC    — remove incremental-capacity feature branch (CTBA intact)

Each ablation is trained from scratch and evaluated against the full model.
Results are printed as a table and optionally saved to a CSV.

Usage:
    py scripts\\ablation.py \\
        --data-dir data/raw/nasa \\
        --dataset nasa \\
        --epochs 200 \\
        --out results/ablation.csv
"""
import argparse
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uapi_former.model import (UAPIFormer, DTAGBlock, EITE, PRAPPooling,
                               DynamicChemistryToken, ICFeatureExtractor,
                               CrossTaskAttentionBlock)
from uapi_former.evidential import nig_loss, EvidentialNIGHead
from uapi_former.metrics import rmse, mae


# ---------------------------------------------------------------------------
# Ablated model variants
# ---------------------------------------------------------------------------

class UAPIFormerNoEITE(UAPIFormer):
    """EITE ablation: zero out SPMe channels so model sees no physics information.

    Explicitly blanks channels 3 (V_spme) and 4 (residual) before the forward
    pass, and passes v_spme=None so PRAP also falls back to uniform mean
    pooling.  Necessary because EITE's zero-pad branch only fires when
    x.size(-1)==3; with the 5-channel dataset x the channels would otherwise
    leak physics information through the direct projection path (Rule 6).
    """
    def forward(self, x, v_spme=None):
        if x.size(-1) >= 5:
            x = x.clone()
            x[..., 3:] = 0.0   # blank V_spme (ch3) and residual (ch4)
        return super().forward(x, v_spme=None)  # v_spme=None also disables PRAP


class _MSEHead(nn.Module):
    """Standard point-estimate regression head (replaces NIG for ablation)."""
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class UAPIFormerNoNIG(nn.Module):
    """NIG ablation: swap NIG heads with plain MSE heads.

    All other components (DCT, EITE, PRAP, DTAG, CTBA, IC) are identical to
    the full model so the ablation isolates the uncertainty head only (Rule 6).
    """
    def __init__(self, in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=200):
        super().__init__()
        from uapi_former.model import PositionalEncoding
        self.dct          = DynamicChemistryToken(in_channels=in_channels, d_model=d_model)
        self.eite         = EITE(in_channels=in_channels, d_model=d_model)
        self.pos_enc      = PositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model * 4,
                                               dropout=0.1, batch_first=True)
        self.encoder      = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.prap         = PRAPPooling()
        self.dtag         = DTAGBlock(d_model)
        self.ic_extractor = ICFeatureExtractor(n_bins=20, d_out=d_model // 4)
        self.ctba         = CrossTaskAttentionBlock(d_model, nhead=nhead)
        self.soc_head     = _MSEHead(d_model)
        self.soh_head     = _MSEHead(d_model + d_model // 4)  # IC augmented

    def forward(self, x, v_spme=None):
        ic_feat = self.ic_extractor(x)
        residual_mag = torch.abs(x[..., 0] - v_spme) if v_spme is not None else None
        chem_token = self.dct(x).unsqueeze(1)
        x = self.eite(x, v_spme)
        x = self.pos_enc(x)
        x = torch.cat([chem_token, x], dim=1)
        x = self.encoder(x)
        rep = self.prap(x[:, 1:, :], residual_mag)
        soc_rep, soh_rep = self.dtag(rep)
        soc_rep, soh_rep = self.ctba(soc_rep, soh_rep)
        soh_rep_full = torch.cat([soh_rep, ic_feat], dim=-1)
        return self.soc_head(soc_rep), self.soh_head(soh_rep_full)


class UAPIFormerNoDTAG(nn.Module):
    """DTAG+CTBA ablation: shared rep fed directly to both heads, no cross-task attention.

    DTAG and CTBA are ablated jointly because CTBA operates on the task-specific
    reps produced by DTAG; without DTAG there is no differentiated signal for
    CTBA to cross-attend.  All other components (DCT, EITE, PRAP, IC) are
    identical to the full model (Rule 6).  The separate No-CTBA row isolates
    CTBA's individual contribution with DTAG intact.
    """
    def __init__(self, in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=200):
        super().__init__()
        from uapi_former.model import PositionalEncoding
        self.dct          = DynamicChemistryToken(in_channels=in_channels, d_model=d_model)
        self.eite         = EITE(in_channels=in_channels, d_model=d_model)
        self.pos_enc      = PositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model * 4,
                                               dropout=0.1, batch_first=True)
        self.encoder      = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.prap         = PRAPPooling()
        self.ic_extractor = ICFeatureExtractor(n_bins=20, d_out=d_model // 4)
        self.soc_head     = EvidentialNIGHead(d_model, out_features=1)
        self.soh_head     = EvidentialNIGHead(d_model + d_model // 4, out_features=1)

    def forward(self, x, v_spme=None):
        ic_feat = self.ic_extractor(x)
        residual_mag = torch.abs(x[..., 0] - v_spme) if v_spme is not None else None
        chem_token = self.dct(x).unsqueeze(1)
        x = self.eite(x, v_spme)
        x = self.pos_enc(x)
        x = torch.cat([chem_token, x], dim=1)
        x = self.encoder(x)
        rep = self.prap(x[:, 1:, :], residual_mag)
        # No DTAG, no CTBA: shared rep fed directly to both heads
        soh_rep_full = torch.cat([rep, ic_feat], dim=-1)
        return self.soc_head(rep), self.soh_head(soh_rep_full)


class UAPIFormerNoPRAP(UAPIFormer):
    """PRAP ablation: uniform mean pooling instead of physics-residual weighting.

    EITE still receives v_spme (physics embedding intact); only the pooling
    step is swapped for mean.  CTBA and IC are preserved to isolate PRAP's
    contribution (Rule 6).
    """
    def forward(self, x, v_spme=None):
        ic_feat = self.ic_extractor(x)
        chem_token = self.dct(x).unsqueeze(1)
        x_emb = self.eite(x, v_spme)
        x_emb = self.pos_enc(x_emb)
        x_enc = torch.cat([chem_token, x_emb], dim=1)
        x_enc = self.encoder(x_enc)
        rep = self.prap(x_enc[:, 1:, :], None)   # None → uniform mean pooling
        soc_rep, soh_rep = self.dtag(rep)
        soc_rep, soh_rep = self.ctba(soc_rep, soh_rep)
        soh_rep_full = torch.cat([soh_rep, ic_feat], dim=-1)
        return self.soc_head(soc_rep), self.soh_head(soh_rep_full)


class UAPIFormerNoDCT(UAPIFormer):
    """DCT ablation: no chemistry fingerprint token; encoder sees T timesteps only.

    EITE, PRAP, CTBA, and IC are preserved to isolate DCT's contribution (Rule 6).
    """
    def forward(self, x, v_spme=None):
        ic_feat = self.ic_extractor(x)
        residual_mag = torch.abs(x[..., 0] - v_spme) if v_spme is not None else None
        x_emb = self.eite(x, v_spme)
        x_emb = self.pos_enc(x_emb)
        x_enc = self.encoder(x_emb)            # (B, T, d_model) — no CLS prepended
        rep = self.prap(x_enc, residual_mag)   # pool over all T timesteps
        soc_rep, soh_rep = self.dtag(rep)
        soc_rep, soh_rep = self.ctba(soc_rep, soh_rep)
        soh_rep_full = torch.cat([soh_rep, ic_feat], dim=-1)
        return self.soc_head(soc_rep), self.soh_head(soh_rep_full)


class UAPIFormerNoCTBA(UAPIFormer):
    """CTBA ablation: skip cross-task bidirectional attention; DTAG intact.

    After DTAG splits the shared rep, the CTBA step is bypassed — soc_rep and
    soh_rep are fed directly to the heads without cross-task interaction.  IC
    features are still concatenated to soh_rep so that only CTBA's contribution
    is measured (Rule 6).
    """
    def forward(self, x, v_spme=None):
        ic_feat = self.ic_extractor(x)
        residual_mag = torch.abs(x[..., 0] - v_spme) if v_spme is not None else None
        chem_token = self.dct(x).unsqueeze(1)
        x_emb = self.eite(x, v_spme)
        x_emb = self.pos_enc(x_emb)
        x_enc = torch.cat([chem_token, x_emb], dim=1)
        x_enc = self.encoder(x_enc)
        rep = self.prap(x_enc[:, 1:, :], residual_mag)
        soc_rep, soh_rep = self.dtag(rep)
        # Skip CTBA
        soh_rep_full = torch.cat([soh_rep, ic_feat], dim=-1)
        return self.soc_head(soc_rep), self.soh_head(soh_rep_full)


class SelfProjectionBlock(CrossTaskAttentionBlock):
    """Parameter-matched CTBA control (revision R2.7).

    Identical modules and parameter count to CrossTaskAttentionBlock, but each
    stream uses ITS OWN vector as key/value (self instead of cross). With
    length-one sequences both variants reduce to a learned projection; the only
    difference is whether cross-task information flows between SOC and SOH.
    """
    def forward(self, soc_rep, soh_rep):
        s = soc_rep.unsqueeze(1)
        h = soh_rep.unsqueeze(1)
        s_x, _ = self.soc_attn(s, s, s)
        s = self.soc_norm1(s + s_x)
        s = self.soc_norm2(s + self.soc_ff(s))
        h_x, _ = self.soh_attn(h, h, h)
        h = self.soh_norm1(h + h_x)
        h = self.soh_norm2(h + self.soh_ff(h))
        return s.squeeze(1), h.squeeze(1)


class UAPIFormerCTBASelf(UAPIFormer):
    """CTBA replaced by its parameter-matched self-projection control."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", 128)
        nhead = kwargs.get("nhead", 4)
        self.ctba = SelfProjectionBlock(d_model, nhead=nhead, dropout=0.1)


class UAPIFormerNoIC(UAPIFormer):
    """IC ablation: no incremental-capacity feature branch; CTBA intact.

    The IC extractor and its concatenation to soh_rep are omitted.  CTBA still
    operates on the DTAG-gated reps, so only the IC branch's contribution is
    measured (Rule 6).  soh_head is re-initialised with d_model inputs (vs
    d_model + d_model//4 in the full model) to match the reduced dimension.
    """
    def __init__(self, in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=200):
        super().__init__(in_channels=in_channels, d_model=d_model, nhead=nhead,
                         num_layers=num_layers, seq_len=seq_len)
        # Override SOH head: no IC concatenation so input is d_model only
        self.soh_head = EvidentialNIGHead(d_model, out_features=1)

    def forward(self, x, v_spme=None):
        residual_mag = torch.abs(x[..., 0] - v_spme) if v_spme is not None else None
        chem_token = self.dct(x).unsqueeze(1)
        x_emb = self.eite(x, v_spme)
        x_emb = self.pos_enc(x_emb)
        x_enc = torch.cat([chem_token, x_emb], dim=1)
        x_enc = self.encoder(x_enc)
        rep = self.prap(x_enc[:, 1:, :], residual_mag)
        soc_rep, soh_rep = self.dtag(rep)
        soc_rep, soh_rep = self.ctba(soc_rep, soh_rep)
        # Skip IC concatenation — pass soh_rep directly
        return self.soc_head(soc_rep), self.soh_head(soh_rep)


# ---------------------------------------------------------------------------
# Generic train/evaluate helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, scaler, device, is_nig=True, mse_only=True):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch; v_spme = None
        else:
            x, v_spme, soc, soh = batch; v_spme = v_spme.to(device)
        x   = x.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            soc_out, soh_out = model(x, v_spme)
            if not is_nig:
                # Plain MSE head (UAPIFormerNoNIG variant)
                loss = 10.0 * F.mse_loss(soc_out, soc) + 2.0 * F.mse_loss(soh_out, soh)
            elif mse_only:
                # NIG model but trained with MSE only (canonical v12-MSE approach)
                sg,sv,sa,sb = soc_out; hg,hv,ha,hb = soh_out
                loss = 10.0 * F.mse_loss(sg, soc) + 2.0 * F.mse_loss(hg, soh)
            else:
                # Full NIG ELBO + MSE anchor
                sg,sv,sa,sb = soc_out; hg,hv,ha,hb = soh_out
                loss = (nig_loss(soc,sg,sv,sa,sb,coeff=0.1)
                        + nig_loss(soh,hg,hv,ha,hb,coeff=0.1)
                        + 10.0 * F.mse_loss(sg, soc)
                        + 2.0  * F.mse_loss(hg, soh))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)


@torch.no_grad()
def _evaluate(model, loader, device, is_nig=True):  # noqa: E302
    model.eval()
    sp, st, hp, ht = [], [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch; v_spme = None
        else:
            x, v_spme, soc, soh = batch; v_spme = v_spme.to(device)
        x = x.to(device)
        soc_out, soh_out = model(x, v_spme)
        if is_nig:
            sp.append(soc_out[0].cpu()); hp.append(soh_out[0].cpu())
        else:
            sp.append(soc_out.cpu()); hp.append(soh_out.cpu())
        st.append(soc); ht.append(soh)
    sp = torch.cat(sp).view(-1); st = torch.cat(st).view(-1)
    hp = torch.cat(hp).view(-1); ht = torch.cat(ht).view(-1)
    return rmse(sp,st), mae(sp,st), rmse(hp,ht), mae(hp,ht)


def run_variant(name, model_cls, args, train_loader, test_loader, device, is_nig=True):
    print(f"\n{'='*55}\n  Ablation: {name}\n{'='*55}")
    model = model_cls().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sc    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    for ep in range(1, args.epochs + 1):
        # All ablation variants use MSE-only loss so comparisons are fair on RMSE
        loss = _train_one_epoch(model, train_loader, opt, sc, device,
                                is_nig=is_nig, mse_only=True)
        sch.step()
        if ep % 50 == 0:
            print(f"  Epoch {ep}/{args.epochs}  loss={loss:.6f}")
    soc_rmse, soc_mae, soh_rmse, soh_mae = _evaluate(model, test_loader, device, is_nig=is_nig)
    print(f"  SOC RMSE={soc_rmse*100:.2f}%  MAE={soc_mae*100:.2f}%")
    print(f"  SOH RMSE={soh_rmse:.4f}       MAE={soh_mae:.4f}")
    return {"name": name, "soc_rmse": soc_rmse, "soc_mae": soc_mae,
            "soh_rmse": soh_rmse, "soh_mae": soh_mae}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=str, default=None)
    p.add_argument("--dataset",    type=str, default="synthetic",
                   choices=["synthetic","nasa","calce","mit_tri","oxford","memmap"])
    p.add_argument("--spme-cache", type=str, default=None)
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seq-len",    type=int, default=200)
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",        type=str, default=None)
    p.add_argument("--num-synthetic", type=int, default=4096)
    p.add_argument("--split-mode",   type=str, default="intra_cell_random",
                   choices=["cross_cell", "intra_cell", "intra_cell_random"])
    p.add_argument("--seed",         type=int, default=None,
                   help="Random seed for reproducibility (torch/cuda/numpy/random). "
                        "When set, seeds are applied before any model/data initialization.")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # load datasets
    def _make_ds(split):
        if args.dataset == "synthetic":
            from uapi_former.dataset import SyntheticDataset
            return SyntheticDataset(num_samples=args.num_synthetic, seq_len=args.seq_len, in_channels=6)
        import uapi_former.dataset as dm
        cls_map = {"nasa":"NASABatteryDataset","calce":"CALCEDataset",
                   "mit_tri":"MITTRIDataset","oxford":"OxfordDataset","memmap":"MemmapDataset"}
        cls = getattr(dm, cls_map[args.dataset])
        if args.dataset == "memmap":
            return cls(args.data_dir)
        kwargs = {"split": split, "seq_len": args.seq_len, "spme_cache": args.spme_cache}
        if args.dataset == "nasa":
            kwargs["split_mode"] = args.split_mode
        return cls(args.data_dir, **kwargs)

    train_ds = _make_ds("train"); test_ds = _make_ds("test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    sl = args.seq_len
    results = []

    # Full model (all 7 novel components)
    results.append(run_variant(
        "Full (EITE+NIG+DTAG+PRAP+DCT+CTBA+IC)",
        lambda: UAPIFormer(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # No-EITE
    results.append(run_variant(
        "No-EITE (SPMe channels zeroed)",
        lambda: UAPIFormerNoEITE(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # No-NIG
    results.append(run_variant(
        "No-NIG (MSE heads)",
        lambda: UAPIFormerNoNIG(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=False))

    # No-DTAG+CTBA (ablated jointly; see docstring)
    results.append(run_variant(
        "No-DTAG+CTBA (shared rep, no cross-task attn)",
        lambda: UAPIFormerNoDTAG(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # No-PRAP
    results.append(run_variant(
        "No-PRAP (uniform mean pool)",
        lambda: UAPIFormerNoPRAP(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # No-DCT
    results.append(run_variant(
        "No-DCT (no chemistry token)",
        lambda: UAPIFormerNoDCT(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # No-CTBA
    results.append(run_variant(
        "No-CTBA (no cross-task attention)",
        lambda: UAPIFormerNoCTBA(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # No-IC
    results.append(run_variant(
        "No-IC (no incremental capacity branch)",
        lambda: UAPIFormerNoIC(in_channels=6, d_model=128, nhead=4, num_layers=4, seq_len=sl),
        args, train_loader, test_loader, device, is_nig=True))

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {'Variant':<35}  SOC RMSE%  SOH RMSE")
    print(f"  {'-'*35}  ---------  --------")
    for r in results:
        print(f"  {r['name']:<35}  {r['soc_rmse']*100:>7.2f}%  {r['soh_rmse']:>8.4f}")
    print(f"{'='*70}")

    if args.out:
        import csv
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

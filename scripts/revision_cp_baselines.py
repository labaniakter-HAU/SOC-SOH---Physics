"""Revision R2.5b: conformal baselines missing from the submitted version.

All three are applied to the SAME clean LOCO folds, with the same frozen
estimator and the same calibration partition, so only the calibration scheme
differs.

  CQR       Conformalized quantile regression (Romano et al. 2019) on the frozen
            128-d pooled latent: two linear quantile heads (alpha/2, 1-alpha/2)
            trained to convergence (full-batch Adam, cosine-annealed learning rate)
            with the pinball loss on the SOURCE TRAINING partition only,
            then conformalized on the calibration partition with the CQR score
            E = max(q_lo - y, y - q_hi).
  Mondrian  Group-conditional split conformal: groups are terciles of the
            model's own PREDICTED SOH (available at test time in a BMS), with
            boundaries fixed on the calibration partition; a separate quantile
            is taken per group.
  ACI       Adaptive conformal inference (Gibbs & Candes 2021) over the target
            stream in cycle order, with labels revealed only when a cycle ends:
            alpha_{t+1} = alpha_t + gamma * (alpha - err_t). Infinite intervals
            (alpha_t <= 0) are counted as refusals, not as coverage. Step-size
            sensitivity gamma in GAMMAS.

Diagnostics added in revision: CQR in-distribution half-split check, median and
range-clipped widths, share of full-range intervals; CQR with MLP heads; SOC > 0
stratum for every scheme.

Writes results/revision/cp_baselines_s{seed}.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from uapi_former.dataset import NASABatteryDataset
from uapi_former.model import UAPIFormer
from scripts.shift_conformal_study import load_model, collect
from uapi_former.conformal_extras import restrict_active

CELLS = ["B0005", "B0006", "B0007", "B0018"]
ALPHA = 0.10
GAMMA = 0.01
GAMMAS = (0.005, 0.01, 0.05)     # ACI step-size sensitivity


def pinball_fit(X, y, taus, epochs=2000, lr=1e-2, seed=0, hidden=0, cosine=True):
    """hidden=0: linear heads (primary). hidden>0: one-hidden-layer MLP heads (sensitivity).
    Full-batch Adam with the learning rate annealed to zero (cosine), so the fit converges.
    The submitted fit (constant lr 0.05, 800 epochs, no annealing) never converged: its
    final pinball loss was about 40 times that of the annealed fit and its intervals were
    over 100 SOC points wide even on source calibration windows. It is kept only as a
    recorded diagnostic (CQR_submitted_fit)."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Xs = (X - mu) / sd
    if hidden:
        lin = torch.nn.Sequential(torch.nn.Linear(Xs.shape[1], hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, len(taus)))
    else:
        lin = torch.nn.Linear(Xs.shape[1], len(taus))
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    t = torch.tensor(taus).view(1, -1)
    for _ in range(epochs):
        opt.zero_grad()
        pred = lin(Xs)
        e = y.view(-1, 1) - pred
        loss = torch.maximum(t * e, (t - 1) * e).mean()
        loss.backward(); opt.step()
        if sch:
            sch.step()
    with torch.no_grad():
        e = y.view(-1, 1) - lin(Xs)
        final = float(torch.maximum(t * e, (t - 1) * e).mean())
    qf = lambda Z: lin((Z - mu) / sd).detach()
    qf.train_pinball = final
    return qf


def cqr(qf, cal, tgt, seed):
    """CQR intervals on the target plus the diagnostics needed to read their width:
    an in-distribution check (conformalize on one random half of the calibration
    windows, evaluate on the other half), the median raw width, the share of target
    intervals that contain the whole physical range [0, 1], the share of inverted
    intervals (upper < lower), and the mean width after intersecting with [0, 1].
    The two predicted quantiles are sorted before conformalization (monotone
    rearrangement): linear heads can cross on shifted inputs, and a crossed pair would
    otherwise give an empty interval. The pre-sort crossing share is recorded."""
    qc, qt = qf(cal["rep"].float()), qf(tgt["rep"].float())
    crossing = float((qt[:, 0] > qt[:, 1]).float().mean())
    qc, qt = qc.sort(dim=1).values, qt.sort(dim=1).values
    score = torch.maximum(qc[:, 0] - cal["soc"], cal["soc"] - qc[:, 1])
    n = len(score)
    Q = float(torch.quantile(score, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    lo, hi = qt[:, 0] - Q, qt[:, 1] + Q
    covered = ((tgt["soc"] >= lo) & (tgt["soc"] <= hi)).numpy()
    res = summarize(covered, (hi - lo).clamp(min=0).numpy())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    A, B = perm[: n // 2], perm[n // 2:]
    nA = len(A)
    QA = float(torch.quantile(score[A], min(1.0, np.ceil((nA + 1) * (1 - ALPHA)) / nA)))
    loB, hiB = qc[B, 0] - QA, qc[B, 1] + QA
    w = (hi - lo).clamp(min=0).numpy()
    res["diag"] = {
        "tgt_crossing_share_presort": crossing,
        "src_half_coverage": float(((cal["soc"][B] >= loB) & (cal["soc"][B] <= hiB)).float().mean()),
        "src_half_width_pp": float(100 * (hiB - loB).clamp(min=0).mean()),
        "tgt_median_width_pp": float(100 * np.median(w)),
        "tgt_full_range_share": float(((lo <= 0) & (hi >= 1)).float().mean()),
        "tgt_inverted_share": float((hi < lo).float().mean()),
        "tgt_clipped_width_pp": float(100 * (torch.clamp(hi, max=1) - torch.clamp(lo, min=0)).clamp(min=0).mean()),
        "Q": Q, "train_pinball": qf.train_pinball}
    return res, covered, np.zeros(len(covered), bool)


def aci(s_np, err_all, sig_all, cyc, gamma):
    """ACI over the target stream in cycle order; labels revealed when a cycle ends."""
    order = sorted(set(cyc), key=lambda k_: int(k_.split(":")[1]))
    alpha_t = ALPHA
    cov, wid, ref = np.zeros(len(cyc), bool), np.full(len(cyc), np.nan), np.zeros(len(cyc), bool)
    for cy in order:
        m = np.where(cyc == cy)[0]
        if alpha_t <= 0:                      # infinite interval
            cov[m] = True; ref[m] = True
            err_rate = 0.0
        elif alpha_t >= 1:                    # empty interval
            cov[m] = False; wid[m] = 0.0
            err_rate = 1.0
        else:
            lev = min(1.0, np.ceil((len(s_np) + 1) * (1 - alpha_t)) / len(s_np))
            q = float(np.quantile(s_np, lev))
            ok = err_all[m] <= q * sig_all[m]
            cov[m] = ok; wid[m] = 2 * q * sig_all[m]
            err_rate = float(1 - ok.mean())
        alpha_t = alpha_t + gamma * (ALPHA - err_rate)
    res = summarize(cov, wid[~ref], ref)
    res["final_alpha"] = float(alpha_t)
    return res, cov, ref


def soc_pos(covered, refused, pos):
    """Accepted coverage and finite-and-correct share over the windows with SOC > 0."""
    c, r = np.asarray(covered, bool)[pos], np.asarray(refused, bool)[pos]
    return {"n": int(pos.sum()),
            "accepted_coverage": float(c[~r].mean()) if (~r).sum() else float("nan"),
            "finite_and_correct": float((c & ~r).mean())}


def summarize(covered, widths, refused=None):
    covered = np.asarray(covered, dtype=float)
    out = {"coverage": float(covered.mean()), "n": int(len(covered)),
           "mean_width_pp": float(100 * np.mean(widths)) if len(widths) else float("nan")}
    if refused is not None:
        r = np.asarray(refused, dtype=bool)
        out["refusal"] = float(r.mean())
        out["accepted_coverage"] = float(covered[~r].mean()) if (~r).sum() else float("nan")
        out["finite_and_correct"] = float((covered.astype(bool) & ~r).mean())
    else:
        out["refusal"] = 0.0
        out["accepted_coverage"] = out["coverage"]
        out["finite_and_correct"] = out["coverage"]
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--active", action="store_true",
                   help="calibrate and evaluate on windows with SOC label > 0 only (spec 2026-09-24 §4.1)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = torch.device(a.device)
    out = {}
    for cell in CELLS:
        c = torch.load(f"results/revision/cache/loco_clean_s{a.seed}_{cell}.pt",
                       map_location="cpu", weights_only=False)
        cal, tgt = c["cal"], c["tgt"]
        cyc = np.array([str(x) for x in c["tgt_cycle"]])
        if a.active:
            cal, tgt, cyc = restrict_active(cal, tgt, cyc)
        ck = f"checkpoints/loco_clean_s{a.seed}_{cell}/best.pt"
        raw = torch.load(ck, map_location="cpu", weights_only=False)
        model, meta = load_model(ck, dev)
        tr_ds = NASABatteryDataset("data/raw/nasa", split="train", split_mode="loco", held_out_cell=cell,
                                   ocv_pts=np.array(raw["ocv_pts"]), q_norm=raw["q_norm"],
                                   loco_split="random3", loco_seed=raw["loco_seed"])
        tr = collect(model, DataLoader(tr_ds, batch_size=256, shuffle=False), dev, meta)
        e = {}

        pos = (tgt["soc"] > 1e-6).numpy()       # SOC > 0 stratum (Sec. 3.1)
        strata = {}

        # ---------------- CQR -------------------------------------------------
        taus = [ALPHA / 2, 1 - ALPHA / 2]
        Xtr, ytr = tr["rep"].float(), tr["soc"].float()
        e["CQR"], cv, rf = cqr(pinball_fit(Xtr, ytr, taus, seed=a.seed), cal, tgt, a.seed)
        strata["CQR"] = soc_pos(cv, rf, pos)
        # sensitivity: MLP quantile heads (128 -> 64 -> 2), same data, schedule and conformalization
        e["CQR_mlp"], cv, rf = cqr(pinball_fit(Xtr, ytr, taus, seed=a.seed, hidden=64), cal, tgt, a.seed)
        strata["CQR_mlp"] = soc_pos(cv, rf, pos)
        # record only: the submitted, non-converged fit (constant lr 0.05, 800 epochs)
        e["CQR_submitted_fit"] = cqr(pinball_fit(Xtr, ytr, taus, epochs=800, lr=0.05, seed=a.seed, cosine=False),
                                     cal, tgt, a.seed)[0]

        # ---------------- Mondrian (predicted-SOH terciles) -------------------
        s_cal, _ = sc.nonconformity(cal["sg"], cal["sv"], cal["sa"], cal["sb"], cal["soc"])
        _, sig_t = sc.nonconformity(tgt["sg"], tgt["sv"], tgt["sa"], tgt["sb"], tgt["soc"])
        edges = np.quantile(cal["hg"].numpy(), [1 / 3, 2 / 3])
        gc = np.digitize(cal["hg"].numpy(), edges)
        gt = np.digitize(tgt["hg"].numpy(), edges)
        covered, widths, refused = np.zeros(len(gt), bool), np.zeros(len(gt)), np.zeros(len(gt), bool)
        for g in (0, 1, 2):
            mt, mc = gt == g, gc == g
            if mt.sum() == 0:
                continue
            if mc.sum() < 20:            # too few calibration points -> refuse group
                refused[mt] = True; covered[mt] = True; continue
            q = sc.vanilla_q(s_cal[torch.as_tensor(mc)], 1 - ALPHA)
            err = torch.abs(tgt["soc"] - tgt["sg"])[torch.as_tensor(mt)]
            covered[mt] = (err <= q * sig_t[torch.as_tensor(mt)]).numpy()
            widths[mt] = (2 * q * sig_t[torch.as_tensor(mt)]).numpy()
        e["Mondrian"] = summarize(covered, widths[~refused], refused)
        e["Mondrian"]["group_sizes_cal"] = [int((gc == g).sum()) for g in (0, 1, 2)]
        e["Mondrian"]["group_sizes_tgt"] = [int((gt == g).sum()) for g in (0, 1, 2)]
        strata["Mondrian"] = soc_pos(covered, refused, pos)

        # ---------------- ACI (online, cycle-delayed feedback) ----------------
        err_all = torch.abs(tgt["soc"] - tgt["sg"]).numpy()
        s_np, sig_np = s_cal.numpy(), sig_t.numpy()
        e["ACI"], cv, rf = aci(s_np, err_all, sig_np, cyc, GAMMA)
        strata["ACI"] = soc_pos(cv, rf, pos)
        e["ACI_gamma"] = {str(g): aci(s_np, err_all, sig_np, cyc, g)[0] for g in GAMMAS}

        # ---------------- references -----------------------------------------
        q0 = sc.vanilla_q(s_cal, 1 - ALPHA)
        ok0 = err_all <= float(q0) * sig_np
        e["S0"] = summarize(ok0, 2 * float(q0) * sig_np)
        strata["S0"] = soc_pos(ok0, np.zeros(len(ok0), bool), pos)
        e["soc_pos"] = strata
        out[cell] = e
        main_ = {k: v for k, v in e.items() if "accepted_coverage" in v}
        print(cell, {k: round(100 * v["accepted_coverage"], 1) for k, v in main_.items()},
              "| widths", {k: round(v["mean_width_pp"], 1) for k, v in main_.items()}, flush=True)
    path = a.out or f"results/revision/cp_baselines_s{a.seed}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

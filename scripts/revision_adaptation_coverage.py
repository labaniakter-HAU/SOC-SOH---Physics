"""SUPERSEDED by scripts/revision_chemistry_coverage.py (see main()); helpers still used.

Revision R1.9: does adaptation restore COVERAGE, or only point accuracy?

The submitted version evaluated adaptation (EOT, fine-tuning) on RMSE only. This
script computes conformal coverage for the adapted models on all three target
chemistries, with clean target-domain calibration:

  target test windows are grouped into cycles (windows of one cycle are
  contiguous and share an identical SOH label), the cycles are split in half at
  random (seed 2026), one half calibrates and the other half is evaluated.

The fine-tuned checkpoints were selected on the target VAL split, so calibrating
on a disjoint half of the TEST cycles keeps calibration independent of model
selection (R2.8).

Writes results/revision/adaptation_coverage.json
"""
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from uapi_former.dataset import CALCEDataset, OxfordDataset, MITTRIDataset
from scripts.shift_conformal_study import load_model, collect

TARGETS = {
    "calce": (CALCEDataset, "data/raw/calce", "checkpoints/calce_ft/best.pt"),
    "oxford": (OxfordDataset, "data/raw/oxford", "checkpoints/oxford_ft/best.pt"),
    "mit_tri": (MITTRIDataset, "data/raw/mit_tri", "checkpoints/mit_tri_ft/best.pt"),
}
SEED = 2026


def cycle_groups(soh):
    """Group ids from runs of identical SOH labels (windows of one cycle)."""
    s = np.asarray(soh)
    return np.concatenate([[0], np.cumsum(np.abs(np.diff(s)) > 1e-12)])


def halves(groups):
    u = np.unique(groups)
    perm = np.random.RandomState(SEED).permutation(len(u))
    cal_ids = set(u[perm[: len(u) // 2]].tolist())
    is_cal = np.array([g in cal_ids for g in groups])
    return is_cal, len(u)


def evaluate(out, task):
    g, v, a, b, y = (out[k] for k in (("sg", "sv", "sa", "sb", "soc") if task == "soc"
                                      else ("hg", "hv", "ha", "hb", "soh")))
    groups = cycle_groups(out["soh"].numpy())
    is_cal, n_cycles = halves(groups)
    s, sigma = sc.nonconformity(g, v, a, b, y)
    m = {}
    for cov in (0.90, 0.95):
        q = sc.vanilla_q(s[torch.as_tensor(is_cal)], cov)
        ev = torch.as_tensor(~is_cal)
        r = sc.interval_metrics(y[ev], g[ev], sigma[ev], q)
        r["mean_width_pp"] = 100 * r["mean_width"]
        m[f"cov{int(cov*100)}"] = r
    m["rmse_pct"] = float(100 * torch.sqrt(((y - g) ** 2).mean()))
    m["n_cycles"] = int(n_cycles)
    m["n_cal"] = int(is_cal.sum())
    m["n_eval"] = int((~is_cal).sum())
    return m


def main():
    # SUPERSEDED (R1.9): the fine-tuned checkpoints are trained with MSE only, so the evidential
    # sigma used by evaluate() is an untrained projection. Table 10 now comes from
    # scripts/revision_chemistry_coverage.py (absolute-residual score). This module is kept only
    # for cycle_groups() and halves(), which that script imports.
    raise SystemExit("superseded: run scripts/revision_chemistry_coverage.py instead")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for name, (cls, ddir, ckpt) in TARGETS.items():
        if not os.path.exists(ckpt):
            print("skip", name, "(no checkpoint)"); continue
        model, meta = load_model(ckpt, dev)
        ds = cls(ddir, split="test")
        o = collect(model, DataLoader(ds, batch_size=256, shuffle=False), dev, meta)
        out[name] = {"checkpoint": ckpt,
                     "soc": evaluate(o, "soc"), "soh": evaluate(o, "soh")}
        for t in ("soc", "soh"):
            e = out[name][t]
            print(f"{name:8s} {t} rmse={e['rmse_pct']:6.2f}% cov90={100*e['cov90']['coverage']:5.1f}% "
                  f"width90={e['cov90']['mean_width_pp']:7.2f}pp cycles={e['n_cycles']}", flush=True)
    json.dump(out, open("results/revision/adaptation_coverage.json", "w"), indent=2)
    print("wrote results/revision/adaptation_coverage.json")


if __name__ == "__main__":
    main()

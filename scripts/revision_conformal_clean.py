"""Revision R2.8: conformal study with calibration data that are never used for
training, early stopping, checkpoint selection, or temperature scaling.

Levels
  random          canonical checkpoint; calibration = random half of the TEST
                  cycles (seed 2026, identical to scripts/revision_soh_labels.py),
                  evaluation = the other half.
  cross_protocol  canonical checkpoint; calibration = same clean test half;
                  targets = B0025-B0028.
  chemistry       canonical checkpoint; calibration = same clean test half;
                  targets = CALCE / Oxford test (zero-shot; EOT coverage is in
                  scripts/revision_chemistry_coverage.py, on the model it was trained for).
  loco_clean      clean LOCO models (scripts/loco_clean.py, --seed); calibration
                  = the source cells' 'cal' cycles (never trained on or used
                  for early stopping); target = the held-out cell.

Every entry adds, for S0/S1/S2 and both tasks: accepted-point coverage and the
fraction of windows that received a finite AND correct interval (R1.8).
Collected outputs are cached in results/revision/cache/ for R2.3/R2.4/R2.5b.
Writes results/revision/conformal_clean.json (merge by key).
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former.dataset import NASABatteryDataset
from uapi_former.eot import decode_from_rep
from scripts.shift_conformal_study import load_model, collect, epistemic_thresholds, run_schemes

from uapi_former import shift_conformal as sc


def semirelaxed_transport_weights_chunked(z_target, z_calib, reg=0.1, clip=20.0, chunk=8192):
    """Memory-bounded equivalent of sc.semirelaxed_transport_weights (same math:
    global-mean-normalised euclidean cost, row softmax, column mass * n)."""
    zt, zc = z_target.float(), z_calib.float()
    tot, cnt = 0.0, 0
    for i in range(0, len(zt), chunk):
        M = torch.cdist(zt[i:i + chunk], zc, p=2); tot += float(M.sum()); cnt += M.numel()
    mean = max(tot / cnt, 1e-8)
    col = torch.zeros(len(zc), dtype=torch.float64)
    for i in range(0, len(zt), chunk):
        M = torch.cdist(zt[i:i + chunk], zc, p=2) / mean
        col += (torch.softmax(-M / reg, dim=1) / len(zt)).sum(0).double()
    return torch.clamp((col * len(zc)).float(), max=clip)


sc.semirelaxed_transport_weights = semirelaxed_transport_weights_chunked

CANON = "checkpoints/nasa_v12_nig/best_calibrated.pt"
OUT = "results/revision/conformal_clean.json"
CACHE = "results/revision/cache"
CAL_SEED = 2026
mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def subset(d, mask):
    m = torch.as_tensor(mask)
    return {k: v[m] for k, v in d.items()}


def clean_random_halves(model, meta, dev):
    ds = NASABatteryDataset("data/raw/nasa", split="test", split_mode="intra_cell_random")
    out = collect(model, mk(ds), dev, meta)
    keys = np.array([f"{c}:{p}" for c, p, _ in ds.meta])
    ukeys = sorted(set(keys.tolist()))
    perm = np.random.RandomState(CAL_SEED).permutation(len(ukeys))
    cal_keys = set(ukeys[k] for k in perm[: len(ukeys) // 2])
    is_cal = np.array([k in cal_keys for k in keys])
    out["cycle"] = np.array(keys, dtype=object)
    cal = subset({k: v for k, v in out.items() if k != "cycle"}, is_cal)
    ev = subset({k: v for k, v in out.items() if k != "cycle"}, ~is_cal)
    return cal, ev, keys[is_cal], keys[~is_cal]


def derive(entry):
    """Add accepted-point coverage and finite-and-correct fraction to every scheme."""
    for task in ("soc", "soh"):
        for lvl, schemes in entry.get(task, {}).items():
            for name, m in schemes.items():
                if not isinstance(m, dict) or "coverage" not in m:
                    continue
                r = m.get("frac_uncertifiable", 0.0) or 0.0
                cov = m["coverage"]
                m["refusal"] = r
                m["accepted_coverage"] = (cov - r) / (1 - r) if r < 1 else float("nan")
                m["finite_and_correct"] = cov - r   # among all kept windows
    return entry


def save_cache(name, cal, tgt, extra=None):
    os.makedirs(CACHE, exist_ok=True)
    torch.save({"cal": cal, "tgt": tgt, **(extra or {})}, os.path.join(CACHE, f"{name}.pt"))


def merge_write(key, payload):
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data[key] = payload
    json.dump(data, open(OUT, "w"), indent=2)
    print("wrote", OUT, key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", required=True, choices=["random", "cross_protocol", "chemistry", "loco_clean"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nig-seed", type=int, default=None,
                   help="SP4: use the five-seed v12-NIG checkpoint checkpoints/nasa_v12_nig_seed{k} instead of "
                        "the canonical one; keys and caches get the suffix _s{k} (random_s{k}, "
                        "cross_protocol_s{k}, chemistry_<ds>_s{k}, protocol_s{k}_<cell>)")
    a = p.parse_args()
    dev = torch.device(a.device)

    if a.level in ("random", "cross_protocol", "chemistry"):
        k = a.nig_seed
        sfx = "" if k is None else f"_s{k}"
        model, meta = load_model(CANON if k is None else f"checkpoints/nasa_v12_nig_seed{k}/best.pt", dev)
        cal, ev, cal_keys, ev_keys = clean_random_halves(model, meta, dev)
        taus = epistemic_thresholds(cal)
        if a.level == "random":
            e = derive(run_schemes(cal, ev, meta, taus))
            e.update({"epi_thresholds": taus, "n_cal_cycles": len(set(cal_keys)), "n_eval_cycles": len(set(ev_keys))})
            save_cache(f"random{sfx}", cal, ev, {"cal_cycle": cal_keys, "tgt_cycle": ev_keys})
            merge_write(f"random{sfx}", e)
        elif a.level == "cross_protocol":
            pay = {}
            for cell in ["B0025", "B0026", "B0027", "B0028"]:
                ds = NASABatteryDataset("data/raw/nasa", cells=[cell])
                tgt = collect(model, mk(ds), dev, meta)
                pay[cell] = derive(run_schemes(cal, tgt, meta, taus))
                save_cache(f"protocol{sfx}_{cell}", cal, tgt,
                           {"tgt_cycle": np.array([f"{c}:{q}" for c, q, _ in ds.meta], dtype=object)})
                print(cell, round(pay[cell]["soc"]["cov90"]["S0"]["coverage"], 4))
            pay["epi_thresholds"] = taus
            merge_write(f"cross_protocol{sfx}", pay)
        else:
            from uapi_former.dataset import CALCEDataset, OxfordDataset
            from uapi_former.eot import OTTransducer
            for name, cls, ddir in (("calce", CALCEDataset, "data/raw/calce"),
                                    ("oxford", OxfordDataset, "data/raw/oxford")):
                tgt = collect(model, mk(cls(ddir, split="test")), dev, meta)
                # zero-shot only. The saved EOT transducers were trained on v12-MSE latents, so
                # applying them to this (v12-NIG) model's latents is invalid; EOT coverage is
                # computed on the v12-MSE lineage by scripts/revision_chemistry_coverage.py.
                pay = {"zero_shot": derive(run_schemes(cal, tgt, meta, taus))}
                pay["epi_thresholds"] = taus
                merge_write(f"chemistry_{name}{sfx}", pay)
    else:
        pay = {}
        for cell in ["B0005", "B0006", "B0007", "B0018"]:
            ck = f"checkpoints/loco_clean_s{a.seed}_{cell}/best.pt"
            raw = torch.load(ck, map_location="cpu")
            model, meta = load_model(ck, dev)
            kw = dict(split_mode="loco", held_out_cell=cell, ocv_pts=np.array(raw["ocv_pts"]),
                      q_norm=raw["q_norm"], loco_split="random3", loco_seed=raw["loco_seed"])
            dcal = NASABatteryDataset("data/raw/nasa", split="cal", **kw)
            dtgt = NASABatteryDataset("data/raw/nasa", split="test", **kw)
            cal, tgt = collect(model, mk(dcal), dev, meta), collect(model, mk(dtgt), dev, meta)
            taus = epistemic_thresholds(cal)
            e = derive(run_schemes(cal, tgt, meta, taus)); e["epi_thresholds"] = taus
            pay[cell] = e
            save_cache(f"loco_clean_s{a.seed}_{cell}", cal, tgt,
                       {"tgt_cycle": np.array([f"{c}:{q}" for c, q, _ in dtgt.meta], dtype=object),
                        "tgt_start": np.array([st for _, _, st in dtgt.meta]),
                        "cal_cycle": np.array([f"{c}:{q}" for c, q, _ in dcal.meta], dtype=object)})
            print(cell, {k: round(v["coverage"], 3) for k, v in e["soc"]["cov90"].items() if "coverage" in v})
        merge_write(f"loco_clean_s{a.seed}", pay)


if __name__ == "__main__":
    main()

"""Table 10 (cross-chemistry accuracy AND coverage) on ONE model lineage (R1.9; SP2 update).

Every row uses the canonical v12-MSE lineage and the constant-width absolute-residual score
|y - gamma| (sigma = 1): every chemistry checkpoint is trained with MSE only, so the v, alpha
and beta rows of its output heads never receive a gradient and its evidential sigma is untrained.

  source      backbone checkpoints/nasa_v12/best.pt. Calibration = the clean half of the
              NASA random-split test cycles (seed 2026; the same windows as
              scripts/revision_conformal_clean.py). Targets = CALCE, Oxford and MIT-TRI test
              sets zero-shot and through the saved EOT transducers
              (checkpoints/eot_transducer_{ds}.pt, trained on this backbone's latents; the
              Oxford and MIT-TRI transducers were trained in SP2 with the fixed seed of
              scripts/eot_transfer.py, Oxford after the discharge-record relabel).
              S0 and S1 (density ratio fitted on this backbone's latents).
  fine_tuned  checkpoints/{calce,oxford,mit_tri}_ft/best.pt. Calibration = a random half of
              the target TEST cycles (seed 2026, cycles recovered from runs of identical SOH
              labels, as in scripts/revision_adaptation_coverage.py); evaluation = the other half.
  --active    calibration and evaluation restricted to windows with SOC label > 0 (spec
              2026-09-24 §4.1). The RMSE-reproduction asserts always use all windows.

Before writing, asserts that the recomputed SOC RMSEs reproduce the logged ones (evaluation
logs of scripts/evaluate.py, results/eot_transfer.json).

Writes results/revision/chemistry_coverage.json (or --out)
"""
import argparse, json, os, re, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from uapi_former.conformal_extras import active_mask, restrict_active
from uapi_former.dataset import CALCEDataset, OxfordDataset, MITTRIDataset
from uapi_former.eot import OTTransducer, decode_from_rep
from scripts.shift_conformal_study import load_model, collect
from scripts.revision_conformal_clean import clean_random_halves
from scripts.revision_adaptation_coverage import cycle_groups, halves
from scripts import revision_sp4_paths as sp

BACKBONE = "checkpoints/nasa_v12/best.pt"
TARGETS = {"calce": (CALCEDataset, "data/raw/calce"),
           "oxford": (OxfordDataset, "data/raw/oxford"),
           "mit_tri": (MITTRIDataset, "data/raw/mit_tri")}
ZERO_SHOT_LOG = {"calce": "logs/_leakfix_campaign/12_calce_zeroshot_eval.txt",
                 "oxford": "logs/revision/sp2/ox_zeroshot.log",
                 "mit_tri": "logs/_mit_tri/01_zeroshot_eval.txt"}
FINE_TUNED_LOG = {"calce": "logs/_leakfix_campaign/14_calce_finetuned_eval.txt",
                  "oxford": "logs/revision/sp2/ox_ft_eval.log",
                  "mit_tri": "logs/_mit_tri/03_finetuned_eval.txt"}
mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def rmse_pct(y, g):
    return float(100 * torch.sqrt(((y - g) ** 2).mean()))


def logged_rmse(path):
    """SOC RMSE (%) from the last 'SOC  RMSE : x' line of an evaluation log."""
    return 100 * float(re.findall(r"SOC\s+RMSE\s*:\s*([0-9.]+)", open(path, encoding="utf-8", errors="replace").read())[-1])


def metrics(y, g, q):
    """Absolute-residual interval gamma +/- q; refusal (q = inf) reported separately."""
    m = sc.interval_metrics(y, g, torch.ones_like(y), q)
    r = m.get("frac_uncertifiable", 0.0) or 0.0
    m["refusal"] = r
    m["accepted_coverage"] = (m["coverage"] - r) / (1 - r) if r < 1 else float("nan")
    m["finite_and_correct"] = m["coverage"] - r
    m["mean_width_pp"] = 100 * m["mean_width"] if m.get("mean_width") == m.get("mean_width") else float("nan")
    return m


def source_entry(cal, tgt):
    """S0 and S1 for SOC and SOH with source calibration and the absolute-residual score."""
    e = {"n": int(len(tgt["soc"])), "n_cal": int(len(cal["soc"])),
         "soc_rmse_pct": rmse_pct(tgt["soc"], tgt["sg"]), "soh_rmse_pct": rmse_pct(tgt["soh"], tgt["hg"]),
         "discriminator_auc": sc.discriminator_auc(cal["rep"], tgt["rep"])}
    w_cal = sc.logistic_density_ratio(cal["rep"], tgt["rep"])
    w_tgt = sc.logistic_density_ratio(cal["rep"], tgt["rep"], query=tgt["rep"])
    e["ess"] = sc.effective_sample_size(w_cal)
    for task, y, g, yc, gc in (("soc", tgt["soc"], tgt["sg"], cal["soc"], cal["sg"]),
                               ("soh", tgt["soh"], tgt["hg"], cal["soh"], cal["hg"])):
        s_cal = torch.abs(yc - gc)
        e[task] = {"S0": metrics(y, g, sc.vanilla_q(s_cal, 0.90)),
                   "S1": metrics(y, g, sc.weighted_q(s_cal, w_cal, 0.90, w_test=w_tgt))}
    return e


def warp(model, tgt, name, dev, seed=None):
    td = torch.load(sp.eot_transducer(name, seed), map_location="cpu")
    T = OTTransducer(); T.load_state_dict(td["state_dict"]); T.to(dev).eval()
    with torch.no_grad():
        rep_w = T(tgt["rep"].to(dev))
        (sg, _, _, _), (hg, _, _, _) = decode_from_rep(model, rep_w, tgt["ic"].to(dev))
    tw = dict(tgt)
    tw["rep"], tw["sg"], tw["hg"] = rep_w.cpu(), sg.view(-1).cpu(), hg.view(-1).cpu()
    return tw, td.get("variant")


def fine_tuned_entry(out, active=False):
    groups = cycle_groups(out["soh"].numpy())
    is_cal, n_cycles = halves(groups)
    act = active_mask(out["soc"]).numpy() if active else np.ones(len(is_cal), bool)
    c, v, a = (torch.as_tensor(x) for x in (is_cal & act, ~is_cal & act, act))
    e = {"n": int(act.sum()), "n_cycles": int(n_cycles), "n_cal_windows": int(c.sum()),
         "n_eval_windows": int(v.sum()),
         "soc_rmse_pct": rmse_pct(out["soc"][a], out["sg"][a]), "soh_rmse_pct": rmse_pct(out["soh"][a], out["hg"][a])}
    for task, y, g in (("soc", out["soc"], out["sg"]), ("soh", out["soh"], out["hg"])):
        s = torch.abs(y - g)
        e[task] = {f"cov{int(100 * lv)}": metrics(y[v], g[v], sc.vanilla_q(s[c], lv)) for lv in (0.90, 0.95)}
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--active", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="SP4 backbone seed: checkpoints/nasa_v12_seed{s} lineage (scripts/revision_sp4_paths.py)")
    a = p.parse_args()
    BACKBONE = sp.backbone(a.seed)
    dev = torch.device(a.device)
    sel = (lambda c, t: restrict_active(c, t)[:2]) if a.active else (lambda c, t: (c, t))
    res = {"backbone": BACKBONE, "score": "absolute residual |y - gamma| (sigma = 1)",
           "active_windows_only": a.active, "source": {}, "fine_tuned": {}}

    model, meta = load_model(BACKBONE, dev)
    cal, _, cal_keys, _ = clean_random_halves(model, meta, dev)
    res["n_cal_windows"], res["n_cal_cycles"] = int(len(cal["soc"])), int(len(set(cal_keys.tolist())))
    eot_pub = json.load(open(sp.eot_json(a.seed)))
    for name, (cls, ddir) in TARGETS.items():
        tgt = collect(model, mk(cls(ddir, split="test")), dev, meta)
        z = rmse_pct(tgt["soc"], tgt["sg"])
        assert abs(z - logged_rmse(sp.zeroshot_log(name, a.seed))) < 0.01, (name, z, logged_rmse(sp.zeroshot_log(name, a.seed)))
        ent = {"zero_shot": source_entry(*sel(cal, tgt))}
        tw, variant = warp(model, tgt, name, dev, a.seed)
        pub = eot_pub[name]["test"]["soc_rmse"]                  # already in percent
        assert variant == eot_pub[name]["selected_variant"], (name, variant)
        assert abs(rmse_pct(tw["soc"], tw["sg"]) - pub) < 0.01, (name, rmse_pct(tw["soc"], tw["sg"]), pub)
        ent["eot"] = source_entry(*sel(cal, tw))
        ent["eot"]["variant"] = variant
        res["source"][name] = ent
        print(name, {k: (round(v["soc_rmse_pct"], 2), round(100 * v["soc"]["S0"]["coverage"], 3),
                         round(100 * v["soc"]["S1"]["refusal"], 3)) for k, v in ent.items()}, flush=True)
        del tgt, tw

    for name, (cls, ddir) in TARGETS.items():
        ck = f"{sp.ft_dir(name, a.seed)}/best.pt"
        m_ft, meta_ft = load_model(ck, dev)
        out = collect(m_ft, mk(cls(ddir, split="test")), dev, meta_ft)
        full = rmse_pct(out["soc"], out["sg"])
        assert abs(full - logged_rmse(sp.ft_log(name, a.seed))) < 0.01, (name, full, logged_rmse(sp.ft_log(name, a.seed)))
        e = fine_tuned_entry(out, a.active)
        e["checkpoint"] = ck
        res["fine_tuned"][name] = e
        print(name, "FT", round(e["soc_rmse_pct"], 2), round(100 * e["soc"]["cov90"]["coverage"], 2),
              round(e["soc"]["cov90"]["mean_width_pp"], 2), flush=True)

    path = a.out or sp.chem_cov_json(a.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=2)
    os.replace(path + ".tmp", path)
    print("wrote", path)


if __name__ == "__main__":
    main()

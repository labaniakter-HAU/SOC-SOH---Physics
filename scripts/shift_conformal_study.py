"""Shift-robust conformal calibration study (spec 2026-07-10, schemes S0-S3).

Per shift level, loads the SAME frozen checkpoint that produced the published
numbers, collects NIG outputs + pooled `rep` latents for the calibration and
target sets, and evaluates:
  S0 vanilla split CP (reproduces baked q_hats), S1 weighted CP (logistic
  density ratio), S2 weighted CP (semi-relaxed OT weights), S3 epistemic gate.

Both S1 and S2 use the exact per-test-point weighted-quantile construction
(Tibshirani et al. 2019): each test point contributes its OWN density-ratio
weight mass before its coverage-quantile is located. A single group-averaged
weight mass was tried first and found to systematically under-correct
tail-region test points (see tests/test_shift_conformal.py); this orchestrator
always evaluates the weight function at both calibration points (S1/S2
calibration weights) and target points (S1/S2 per-test-point w_test).

Usage:
  py scripts/shift_conformal_study.py --level loco --cells B0006          # smoke
  py scripts/shift_conformal_study.py --level loco                        # 4 folds
  py scripts/shift_conformal_study.py --level cross_protocol
  py scripts/shift_conformal_study.py --level chemistry --dataset calce
Writes/updates results/shift_conformal_study.json (merge-by-key, never
touching other levels' entries). Smoke runs write to
results/_shift_conformal_smoke.json instead (rule: smoke numbers never enter
manuscript artifacts).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from uapi_former.model import UAPIFormer
from uapi_former.eot import encode_to_rep, decode_from_rep
from uapi_former import shift_conformal as sc

LOCO_CELLS = ["B0005", "B0006", "B0007", "B0018"]
COVERAGES = (0.90, 0.95)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("args", {})
    model = UAPIFormer(in_channels=saved.get("in_channels", 6),
                       d_model=saved.get("d_model", 128),
                       nhead=saved.get("nhead", 4),
                       num_layers=saved.get("num_layers", 4),
                       seq_len=saved.get("seq_len", 200))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt.get("calibration", {})


@torch.no_grad()
def collect(model, loader, device, calib_meta):
    """NIG outputs (via the SAME rep tap used everywhere) + latents + ic_feat.

    Deliberately does NOT apply calib_meta's beta_scale: uapi_former/eval.py's
    evaluate_model() (which produced every published coverage_90/coverage_95
    number) and scripts/calibrate.py's _conformal_q() both compute the
    conformal sigma from RAW (v, alpha, beta) -- beta_scale is a separate,
    ECE-only calibration axis in this codebase that is never applied before
    computing conformal q_hat or evaluating coverage. Applying it here would
    silently break byte-identical S0 reproduction against the baked q_hat.
    """
    out = {k: [] for k in ("sg", "sv", "sa", "sb", "hg", "hv", "ha", "hb",
                           "soc", "soh", "rep", "ic")}
    for batch in loader:
        if len(batch) == 3:
            x, soc, soh = batch; v_spme = None
        else:
            x, v_spme, soc, soh = batch; v_spme = v_spme.to(device)
        x = x.to(device)
        rep, ic_feat = encode_to_rep(model, x, v_spme)
        (sg, sv, sa, sb), (hg, hv, ha, hb) = decode_from_rep(model, rep, ic_feat)
        for k, v in zip(("sg", "sv", "sa", "sb", "hg", "hv", "ha", "hb"),
                        (sg, sv, sa, sb, hg, hv, ha, hb)):
            out[k].append(v.cpu())
        out["soc"].append(soc); out["soh"].append(soh)
        out["rep"].append(rep.cpu()); out["ic"].append(ic_feat.cpu())
    res = {k: torch.cat(v) for k, v in out.items()}
    for k in ("sg", "sv", "sa", "sb", "hg", "hv", "ha", "hb", "soc", "soh"):
        res[k] = res[k].view(-1)
    return res


def epistemic_thresholds(cal):
    """Per-task abstention thresholds from SOURCE calibration epistemic
    quantiles only (SOC and SOH epistemic scales differ -- never share)."""
    out = {}
    for task, (v, a, b) in {"soc": (cal["sv"], cal["sa"], cal["sb"]),
                            "soh": (cal["hv"], cal["ha"], cal["hb"])}.items():
        epi = sc.epistemic_variance(v, a, b)
        out[task] = {f"p{p}": float(torch.quantile(epi, p / 100.0))
                     for p in (90, 95, 99)}
    return out


def run_schemes(cal, tgt, calib_meta, epi_thresholds):
    """All schemes for one (calibration set, target set) pair. Returns dict.

    S1/S2 weights are evaluated twice each: once AT calibration points (the
    weighted_q calibration-side `weights`) and once AT target points (the
    per-test-point `w_test`), via logistic_density_ratio(..., query=...) and
    semirelaxed_transport_weights respectively. This is the exact
    per-test-point construction validated in tests/test_shift_conformal.py --
    a single group-averaged weight mass under-corrects tail-region points.
    """
    entry = {"n_calib": len(cal["soc"]), "n_test": len(tgt["soc"])}
    entry["discriminator_auc"] = sc.discriminator_auc(cal["rep"], tgt["rep"])

    w1_cal = sc.logistic_density_ratio(cal["rep"], tgt["rep"])
    w1_tgt = sc.logistic_density_ratio(cal["rep"], tgt["rep"], query=tgt["rep"])
    entry["ess"] = {"S1": sc.effective_sample_size(w1_cal)}

    # S2: semi-relaxed OT weights are inherently defined w.r.t. a query cloud
    # (the "target" side of the transport problem), giving one calibration-side
    # weight vector rather than a natural per-test-point evaluator. S2 therefore
    # uses the group-level w_test approximation (documented coarse scheme); S1
    # is the scheme with the exact per-test-point coverage guarantee.
    w2_cal = sc.semirelaxed_transport_weights(tgt["rep"], cal["rep"])
    entry["ess"]["S2"] = sc.effective_sample_size(w2_cal)

    for task, (g, v, a, b, y) in {
        "soc": (tgt["sg"], tgt["sv"], tgt["sa"], tgt["sb"], tgt["soc"]),
        "soh": (tgt["hg"], tgt["hv"], tgt["ha"], tgt["hb"], tgt["soh"]),
    }.items():
        cg, cv, ca, cb, cy = {
            "soc": (cal["sg"], cal["sv"], cal["sa"], cal["sb"], cal["soc"]),
            "soh": (cal["hg"], cal["hv"], cal["ha"], cal["hb"], cal["soh"]),
        }[task]
        s_cal, _ = sc.nonconformity(cg, cv, ca, cb, cy)
        _, sigma_t = sc.nonconformity(g, v, a, b, y)
        epi_t = sc.epistemic_variance(v, a, b)

        entry.setdefault(task, {})
        for cov in COVERAGES:
            pct = int(cov * 100)
            q_s0 = sc.vanilla_q(s_cal, cov)
            q_s1 = sc.weighted_q(s_cal, w1_cal, cov, w_test=w1_tgt)     # per-test-point, exact
            q_s2 = sc.weighted_q(s_cal, w2_cal, cov)                     # group-level approx
            qs = {"S0": q_s0, "S1": q_s1, "S2": q_s2}
            baked = calib_meta.get(f"conformal_q_{task}_{pct}")
            if baked is not None:
                qs["S0_baked"] = float(baked)   # byte-identity cross-check

            entry[task].setdefault(f"cov{pct}", {})
            for name, q in qs.items():
                m = sc.interval_metrics(y, g, sigma_t, q)
                entry[task][f"cov{pct}"][name] = m

            # S3: epistemic gate on top of one corrected scheme, selected by
            # larger ESS (label-free -- never selected by target coverage).
            best = "S1" if entry["ess"]["S1"] >= entry["ess"]["S2"] else "S2"
            for tau_name, tau in epi_thresholds[task].items():
                keep = epi_t <= tau
                m = sc.interval_metrics(y, g, sigma_t, qs[best], keep=keep)
                m["gated_scheme"] = best
                entry[task][f"cov{pct}"][f"S3_{tau_name}"] = m
    return entry


def merge_write(path, key, payload):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[key] = payload
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path} [{key}]")


def build_loaders_loco(cell, args):
    from uapi_former.dataset import NASABatteryDataset
    cal_ds = NASABatteryDataset(args.data_dir, split="val", split_mode="loco",
                                held_out_cell=cell)
    tgt_ds = NASABatteryDataset(args.data_dir, split="test", split_mode="loco",
                                held_out_cell=cell)
    mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    return mk(cal_ds), mk(tgt_ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", required=True,
                   choices=["loco", "cross_protocol", "chemistry", "random"])
    p.add_argument("--cells", nargs="*", default=None,
                   help="Subset of LOCO/protocol cells (smoke test).")
    p.add_argument("--dataset", default=None, choices=["calce", "oxford"],
                   help="chemistry level only")
    p.add_argument("--transducer", default=None,
                   help="chemistry level: path to saved EOT transducer .pt for the "
                        "EOT-warped S2 variant (omit to skip that variant)")
    p.add_argument("--finetuned-checkpoint", default=None,
                   help="chemistry level: fine-tuned checkpoint for the "
                        "exchangeable upper-bound entry")
    p.add_argument("--data-dir", default="data/raw/nasa")
    p.add_argument("--smoke", action="store_true",
                   help="write to results/_shift_conformal_smoke.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    out_path = ("results/_shift_conformal_smoke.json" if args.smoke
                else "results/shift_conformal_study.json")

    if args.level == "loco":
        cells = args.cells or LOCO_CELLS
        payload = {}
        for cell in cells:
            model, meta = load_model(f"checkpoints/loco_{cell}/best_calibrated.pt", device)
            cal_loader, tgt_loader = build_loaders_loco(cell, args)
            cal = collect(model, cal_loader, device, meta)
            tgt = collect(model, tgt_loader, device, meta)
            taus = epistemic_thresholds(cal)
            payload[cell] = run_schemes(cal, tgt, meta, taus)
            payload[cell]["epi_thresholds"] = taus
            print(cell, "soc cov90:",
                  {k: round(v["coverage"], 3)
                   for k, v in payload[cell]["soc"]["cov90"].items()})
        merge_write(out_path, "loco", payload)

    elif args.level in ("cross_protocol", "random"):
        from uapi_former.dataset import NASABatteryDataset
        model, meta = load_model("checkpoints/nasa_v12_nig/best_calibrated.pt", device)
        cal_ds = NASABatteryDataset(args.data_dir, split="val",
                                    split_mode="intra_cell_random")
        mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        cal = collect(model, mk(cal_ds), device, meta)
        taus = epistemic_thresholds(cal)
        if args.level == "random":
            tgt_ds = NASABatteryDataset(args.data_dir, split="test",
                                        split_mode="intra_cell_random")
            tgt = collect(model, mk(tgt_ds), device, meta)
            payload = run_schemes(cal, tgt, meta, taus)
            payload["epi_thresholds"] = taus
            merge_write(out_path, "random", payload)
        else:
            payload = {}
            cells = args.cells or ["B0025", "B0026", "B0027", "B0028"]
            for cell in cells:
                # B0025-28 are outside CELLS_ALL for split_mode="loco" (that mode
                # only spans B0005/06/07/18); cells=[cell] loads the whole file
                # via the same _load_cell(..., 0.0, 1.0) path (verified identical
                # to scripts/cross_protocol_eval.py's evaluate_cell construction).
                tgt_ds = NASABatteryDataset(args.data_dir, cells=[cell])
                tgt = collect(model, mk(tgt_ds), device, meta)
                payload[cell] = run_schemes(cal, tgt, meta, taus)
            payload["epi_thresholds"] = taus
            merge_write(out_path, "cross_protocol", payload)

    elif args.level == "chemistry":
        from uapi_former.dataset import CALCEDataset, OxfordDataset, NASABatteryDataset
        from uapi_former.eot import OTTransducer
        assert args.dataset, "--dataset calce|oxford required"
        ds_cls = {"calce": CALCEDataset, "oxford": OxfordDataset}[args.dataset]
        ds_dir = {"calce": "data/raw/calce", "oxford": "data/raw/oxford"}[args.dataset]
        model, meta = load_model("checkpoints/nasa_v12_nig/best_calibrated.pt", device)
        mk = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        cal = collect(model, mk(NASABatteryDataset(args.data_dir, split="val",
                                split_mode="intra_cell_random")), device, meta)
        taus = epistemic_thresholds(cal)
        tgt = collect(model, mk(ds_cls(ds_dir, split="test")), device, meta)
        payload = {"zero_shot": run_schemes(cal, tgt, meta, taus)}

        if args.transducer:
            td = torch.load(args.transducer, map_location="cpu")
            T = OTTransducer(); T.load_state_dict(td["state_dict"]); T.to(device).eval()
            with torch.no_grad():
                rep_w = T(tgt["rep"].to(device))
                (sg, sv, sa, sb), (hg, hv, ha, hb) = decode_from_rep(
                    model, rep_w, tgt["ic"].to(device))
            tgt_w = dict(tgt)
            tgt_w["rep"] = rep_w.cpu()
            for k, v in zip(("sg", "sv", "sa", "sb", "hg", "hv", "ha", "hb"),
                            (sg, sv, sa, sb, hg, hv, ha, hb)):
                tgt_w[k] = v.view(-1).cpu()
            # beta_scale deliberately not applied here either -- see collect()'s
            # docstring; conformal sigma is always computed from raw NIG beta.
            payload["eot_warped"] = run_schemes(cal, tgt_w, meta, taus)

        if args.finetuned_checkpoint:
            ft_model, ft_meta = load_model(args.finetuned_checkpoint, device)
            missing = set(model.state_dict()) - set(
                torch.load(args.finetuned_checkpoint, map_location="cpu")["model_state"])
            nig_head_keys = [k for k in missing if "soc_head" in k or "soh_head" in k]
            if nig_head_keys:
                payload["fine_tuned"] = "no_nig_head_not_applicable"
            else:
                ft_cal_ds = ds_cls(ds_dir, split="val")
                ft_cal = collect(ft_model, mk(ft_cal_ds), device, ft_meta)
                ft_tgt = collect(ft_model, mk(ds_cls(ds_dir, split="test")), device, ft_meta)
                s_cal_soc, _ = sc.nonconformity(ft_cal["sg"], ft_cal["sv"], ft_cal["sa"],
                                                ft_cal["sb"], ft_cal["soc"])
                _, sigma_soc = sc.nonconformity(ft_tgt["sg"], ft_tgt["sv"], ft_tgt["sa"],
                                                ft_tgt["sb"], ft_tgt["soc"])
                s_cal_soh, _ = sc.nonconformity(ft_cal["hg"], ft_cal["hv"], ft_cal["ha"],
                                                ft_cal["hb"], ft_cal["soh"])
                _, sigma_soh = sc.nonconformity(ft_tgt["hg"], ft_tgt["hv"], ft_tgt["ha"],
                                                ft_tgt["hb"], ft_tgt["soh"])
                ft_entry = {"note": "exchangeable (target-domain calibration), "
                                    "S1/S2 corrections not applicable"}
                for task, s_cal, g, sigma_t, y in (
                    ("soc", s_cal_soc, ft_tgt["sg"], sigma_soc, ft_tgt["soc"]),
                    ("soh", s_cal_soh, ft_tgt["hg"], sigma_soh, ft_tgt["soh"]),
                ):
                    ft_entry[task] = {}
                    for cov in COVERAGES:
                        q = sc.vanilla_q(s_cal, cov)
                        ft_entry[task][f"cov{int(cov*100)}"] = sc.interval_metrics(y, g, sigma_t, q)
                payload["fine_tuned"] = ft_entry

        payload["epi_thresholds"] = taus
        merge_write(out_path, f"chemistry_{args.dataset}", payload)


if __name__ == "__main__":
    main()

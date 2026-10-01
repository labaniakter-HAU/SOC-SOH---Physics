"""Revision figures, regenerated from the clean-protocol results.

figs/rev_shift_coverage.pdf  accepted-point SOC coverage and S1 refusal by shift
                             level, under the independent-calibration protocol
figs/rev_accuracy.pdf        SOC RMSE across evaluation and adaptation settings,
                             clean LOCO folds (mean +/- SD over seeds)

Reads results/revision/conformal_clean.json (LOCO models of every seed; the five v12-NIG seed
checkpoints for the other levels), loco_clean_s*.json, the five-seed cross-protocol accuracy and
the five backbone lineages of the cross-chemistry chain, so the figures cannot disagree with the
tables. Every bar is a mean over five seeds with its SD.
"""
import glob, json, os, statistics as st

from revision_seeds import SEEDS, seed_files, seed_keys
import revision_sp4_paths as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
CELLS = ["B0005", "B0006", "B0007", "B0018"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in"})


def clean():
    return json.load(open(os.path.join(ROOT, "results/revision/conformal_clean.json")))


def loco_runs():
    return [json.load(open(f)) for f in seed_files("results/revision/loco_clean_s{s}.json", root=ROOT)]


def agg(entries, path):
    vals = []
    for e in entries:
        v = e
        for k in path:
            v = v[k]
        if v == v and abs(v) != float("inf"):
            vals.append(v)
    if not vals:
        return float("nan"), 0.0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def fig_coverage():
    d = clean()
    nig = lambda lv: [d[f"{lv}_s{k}"] for k in SEEDS]
    levels = [("Random", nig("random")),
              *[(c.replace("B00", "B"), [d[k][c] for k in seed_keys(d, "loco_clean_s")]) for c in CELLS],
              *[(c.replace("B00", "B"), [e[c] for e in nig("cross_protocol")]) for c in ["B0025", "B0026", "B0027", "B0028"]],
              ("CALCE", [e["zero_shot"] for e in nig("chemistry_calce")]),
              ("Oxford", [e["zero_shot"] for e in nig("chemistry_oxford")])]
    x = np.arange(len(levels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    for i, (key, lab, col) in enumerate([("S0", "S0 split conformal", "#B0BEC5"),
                                         ("S1", "S1 weighted (accepted)", "#1E88E5")]):
        field = "coverage" if key == "S0" else "accepted_coverage"
        m = [100 * agg(e, ("soc", "cov90", key, field))[0] for _, e in levels]
        s = [100 * agg(e, ("soc", "cov90", key, field))[1] for _, e in levels]
        ref_i = [100 * agg(e, ("soc", "cov90", "S1", "refusal"))[0] for _, e in levels]
        hatch = ["//" if (key == "S1" and r > 50) else "" for r in ref_i]
        bars_i = ax1.bar(x + (i - 0.5) * 0.4, m, 0.38, yerr=s, capsize=1.5, label=lab, color=col,
                         edgecolor="black", linewidth=0.4, error_kw={"lw": 0.6})
        for b, h in zip(bars_i, hatch):
            b.set_hatch(h)
    ax1.axhline(90, ls="--", lw=0.8, color="crimson")
    ax1.text(len(levels) - 0.4, 84, "90% nominal", color="crimson", fontsize=7, ha="right")
    ax1.set_ylabel("SOC coverage (%)"); ax1.set_ylim(0, 132)
    ax1.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               handles=ax1.get_legend_handles_labels()[0] + [plt.Rectangle((0,0),1,1, facecolor="#1E88E5", hatch="//", edgecolor="black", lw=0.4)],
               labels=ax1.get_legend_handles_labels()[1] + [">50% refused"])
    ref = [100 * agg(e, ("soc", "cov90", "S1", "refusal"))[0] for _, e in levels]
    ref_sd = [100 * agg(e, ("soc", "cov90", "S1", "refusal"))[1] for _, e in levels]
    ax2.bar(x, ref, 0.55, yerr=ref_sd, capsize=1.5, color="#6D4C41", edgecolor="black", linewidth=0.4,
            error_kw={"lw": 0.6})
    ax2.set_ylabel("S1 refusal (%)"); ax2.set_ylim(0, 108)
    ax2.set_xticks(x); ax2.set_xticklabels([l for l, _ in levels], rotation=45, ha="right")
    for ax in (ax1, ax2):
        ax.axvspan(0.5, 4.5, color="#1E88E5", alpha=0.05)
        ax.axvspan(4.5, 8.5, color="#43A047", alpha=0.05)
        ax.axvspan(8.5, 10.5, color="#E53935", alpha=0.05)
    ax1.text(2.5, 108, "cross-cell", ha="center", fontsize=7, color="#1E88E5")
    ax1.text(6.5, 108, "cross-protocol", ha="center", fontsize=7, color="#43A047")
    ax1.text(9.5, 108, "cross-chem.", ha="center", fontsize=7, color="#E53935")
    fig.tight_layout()
    out = os.path.join(FIGS, "rev_shift_coverage.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_accuracy():
    runs = loco_runs()
    chem = [json.load(open(os.path.join(ROOT, sp.chem_cov_json(s)))) for s in SEEDS]
    full = [100 * e["soc_rmse"] for e in json.load(open(os.path.join(ROOT, "results/_multiseed_raw/full.json")))]
    bars = [("Random\nsplit", st.mean(full), st.stdev(full), "#1E88E5")]
    for c in CELLS:
        v = [r[c]["soc_rmse"] for r in runs if c in r]
        bars.append((f"LOCO\n{c.replace('B00','B')}", st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), "#43A047"))
    # cross-protocol: per v12-NIG seed the mean over the four cells, then mean +- SD over seeds
    proto = json.load(open(os.path.join(ROOT, "results/revision/seeds/cross_protocol_acc.json")))
    pv = [st.mean(proto[f"s{k}"][c]["soc_rmse"] for c in ["B0025", "B0026", "B0027", "B0028"]) for k in SEEDS]
    bars.append(("Cross-\nprotocol", st.mean(pv), st.stdev(pv), "#FB8C00"))
    import re
    zs = [100 * float(re.findall(r"SOC\s+RMSE\s*:\s*([0-9.]+)", open(os.path.join(ROOT, sp.zeroshot_log("calce", s)),
                                  encoding="utf-8", errors="replace").read())[-1]) for s in SEEDS]
    bars.append(("CALCE\nzero-shot", st.mean(zs), st.stdev(zs), "#E53935"))
    ft = [c["fine_tuned"]["calce"]["soc_rmse_pct"] for c in chem]
    bars.append(("CALCE\nfine-tuned", st.mean(ft), st.stdev(ft), "#8E24AA"))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], 0.62, yerr=[b[2] for b in bars], capsize=2,
           color=[b[3] for b in bars], edgecolor="black", linewidth=0.4, error_kw={"lw": 0.6})
    for xi, b in zip(x, bars):
        ax.text(xi, b[1] + b[2] + 0.6, f"{b[1]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.set_ylabel("SOC RMSE (%)")
    ax.set_ylim(0, max(b[1] for b in bars) * 1.25)
    ax.text(0.99, 0.95, "bars differ in training provenance,\nnot only in shift severity",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, style="italic")
    fig.tight_layout()
    out = os.path.join(FIGS, "rev_accuracy.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_coverage()
    fig_accuracy()

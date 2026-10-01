"""
Regenerate Figure 5 (multi-seed ablation: change in mean SOC RMSE after
component removal) with row labels that match Table 10's current
terminology.

The image previously embedded as Figure 5 has no reproducing script left in
this repo (plot_results.py's plot_ablation_bar_multiseed() produces a
differently-styled chart with value/n annotations and a colormap, which does
not match what's actually embedded) -- but the underlying per-seed data is
still exactly the multi-seed ablation data reported in Table 10
(results/_multiseed_raw/*.json). This script reproduces the plain, single-
colour horizontal-bar style of the currently-embedded figure from that real
data, using Table 10's renamed labels ("Scalar MSE head", "Shared
representation", "No IC-inspired proxy") instead of the old "No NIG",
"No DTAG+CTBA", "No IC", so the figure and table no longer disagree.

No new experiment: the means/stds/p-values are unchanged and were already
independently verified against results/_multiseed_raw/*.json to match
Table 10 exactly.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "figs")
RAW = os.path.join(BASE, "results", "_multiseed_raw")

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FILES = {
    "Full":         "full.json",
    "No-EITE":      "no_eite.json",
    "No-NIG":       "no_nig.json",
    "No-DTAG+CTBA": "no_dtag_ctba.json",
    "No-PRAP":      "no_prap.json",
    "No-DCT":       "no_dct.json",
    "No-CTBA":      "no_ctba.json",
    "No-IC":        "no_ic.json",
}
# p-values are computed below from the raw per-seed files (paired on the seeds each
# variant shares with Full). They used to be hard-coded from a table built on
# values rounded to three decimals, which shifted several in the third decimal.
# Table-10-consistent display labels (Round-3 critique fix).
LABEL_MAP = {
    "No-EITE":      "No EITE",
    "No-NIG":       "Scalar MSE head",
    "No-DTAG+CTBA": "Shared representation",
    "No-PRAP":      "No PRAP",
    "No-DCT":       "No DCT",
    "No-CTBA":      "No CTBA",
    "No-IC":        "No IC-inspired proxy",
}
VARIANTS = ["No-EITE", "No-NIG", "No-DTAG+CTBA", "No-PRAP",
            "No-DCT", "No-CTBA", "No-IC"]


def build():
    from scipy import stats as sps
    runs = {name: {r["seed"]: r["soc_rmse"] * 100 for r in json.load(open(os.path.join(RAW, fname)))}
            for name, fname in FILES.items()}
    stats, PVALUES = {}, {}
    for v in VARIANTS:
        # every quantity for a variant uses only the seeds it shares with Full
        seeds = sorted(set(runs[v]) & set(runs["Full"]))
        var = np.array([runs[v][s] for s in seeds]); ful = np.array([runs["Full"][s] for s in seeds])
        stats[v] = (float((var - ful).mean()), float((var - ful).std(ddof=1)))   # paired
        PVALUES[v] = float(sps.ttest_rel(var, ful).pvalue)
    print("paired p (SOC):", {v: round(p, 4) for v, p in PVALUES.items()})

    labels = [LABEL_MAP[v] for v in VARIANTS]
    delta = [stats[v][0] for v in VARIANTS]
    err = [stats[v][1] for v in VARIANTS]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.48, 4.0))
        bars = ax.barh(range(len(labels)), delta, xerr=err,
                        color="#1f77b4", edgecolor="black", linewidth=0.7,
                        height=0.6,
                        error_kw=dict(elinewidth=1.1, capsize=3.5, ecolor="black"))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Change in mean SOC RMSE after component removal (percentage points)")
        ax.grid(axis="x", color="#cccccc", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

        max_d = max(d + e for d, e in zip(delta, err))
        ax.set_xlim(left=-0.05 - 0.15 * max_d, right=max_d * 1.25)

        for bar, d, e, v in zip(bars, delta, err, VARIANTS):
            sig = "*" if PVALUES[v] < 0.05 else "n.s."
            ax.text(d + e + max_d * 0.03, bar.get_y() + bar.get_height() / 2,
                    sig, va="center", fontsize=10)

        plt.tight_layout(pad=0.4)
        pdf_path = os.path.join(FIGS, "ablation_bar.pdf")
        png_path = os.path.join(FIGS, "ablation_bar.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {pdf_path}")
        print(f"[OK] {png_path}")


if __name__ == "__main__":
    build()

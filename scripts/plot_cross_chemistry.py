"""
Figure 4 (cross-chemistry SOC RMSE: zero-shot, frozen-backbone EOT, fine-tuning), read from
the result files behind Table 6, over the five v12-MSE backbone lineages: zero-shot from each
backbone's evaluation logs and EOT from its eot_transfer JSON (both via
revision_make_tables.cross_chemistry_rmse_seeds), fine-tuned from its chemistry-coverage JSON.
Each bar is the mean over the five backbones and the dots are the individual backbones (an SD
error bar cannot be drawn on the log axis where the spread exceeds the mean, as for Oxford EOT).
An EOT bar whose mean SOC correlation over the backbones is negative is highlighted and annotated
with the mean r and the number of backbones with r < 0: its lower RMSE does not come with SOC tracking.
"""
import json
import os
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "figs")
from revision_make_tables import cross_chemistry_rmse_seeds
from revision_seeds import SEEDS
import revision_sp4_paths as sp

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
YLIM = (0.15, 260)


def bar_values():
    """(label, mean SOC RMSE %, per-backbone values, note) for each bar; nothing typed in."""
    rs = cross_chemistry_rmse_seeds()
    cc = [json.load(open(os.path.join(BASE, sp.chem_cov_json(s)))) for s in SEEDS]
    eot = [json.load(open(os.path.join(BASE, sp.eot_json(s)))) for s in SEEDS]
    out = []
    for k, lab in (("calce", "CALCE"), ("oxford", "Oxford"), ("mit_tri", "MIT-TRI")):
        r = [e[k]["test"]["soc_r"] for e in eot]
        note = f"mean r = {st.mean(r):.3f}\n(r < 0 for {sum(x < 0 for x in r)} of {len(r)} backbones)"
        for stage, vals, nt in (("zero-shot", [x[(k, "zero")] for x in rs], None),
                                ("EOT", [x[(k, "eot")] for x in rs], note if st.mean(r) < 0 else None),
                                ("fine-tuned", [c["fine_tuned"][k]["soc_rmse_pct"] for c in cc], None)):
            out.append((f"{lab}\n{stage}", st.mean(vals), vals, nt))
    allv = [x for _, _, v, _ in out for x in v]
    assert YLIM[0] < min(allv) and max(allv) * 1.6 < YLIM[1], allv   # fixed log axis, room for the labels
    return out


def build():
    BARS = bar_values()
    labels = [b[0] for b in BARS]
    means = [b[1] for b in BARS]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.9, 4.0))
        bars = ax.bar(range(len(labels)), means, color="#1f77b4", edgecolor="black", linewidth=0.6, width=0.62)
        for i, (_, _, vals, _) in enumerate(BARS):
            n = len(vals)
            xs = [i + 0.22 * (j - (n - 1) / 2) / ((n - 1) / 2) for j in range(n)]
            ax.scatter(xs, vals, s=14, color="white", edgecolor="black", linewidths=0.7, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("SOC RMSE (%)")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, (label, v, vals, note) in zip(bars, BARS):
            ax.text(bar.get_x() + bar.get_width() / 2, max(vals) * 1.15,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
            if note is not None:
                bar.set_facecolor("#d73027")
                ax.annotate(
                    note,
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(bar.get_x() + bar.get_width() / 2 + 0.55, v * 3.2),
                    fontsize=8, color="#a50026", fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="#a50026", lw=1.1),
                )

        plt.tight_layout()
        pdf_path = os.path.join(FIGS, "cross_chemistry.pdf")
        png_path = os.path.join(FIGS, "cross_chemistry.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {pdf_path}")
        print(f"[OK] {png_path}")


if __name__ == "__main__":
    os.chdir(BASE)
    build()

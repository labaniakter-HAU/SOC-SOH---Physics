"""Figure S3: sensitivity of the frozen v12-MSE backbones to the fixed R0, mean +- SD over the five
backbones (results/revision/seeds/r0_sensitivity_s{k}.json, scripts/revision_sp4_r0.py, whose
procedure first reproduces the former single-checkpoint sweep results/r0_sensitivity.json).

Usage:
    python scripts/plot_resistance_sensitivity.py
Writes figs/resistance_sensitivity.png (+ .pdf) at 600 dpi.
"""
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_seeds import SEEDS  # noqa: E402

OUT_PNG = os.path.join(ROOT, "figs", "resistance_sensitivity.png")
OUT_PDF = os.path.join(ROOT, "figs", "resistance_sensitivity.pdf")
ORDER = ["-30%", "-20%", "-10%", "0% (baseline)", "+10%", "+20%", "+30%"]


def main():
    runs = [json.load(open(os.path.join(ROOT, f"results/revision/seeds/r0_sensitivity_s{k}.json")))["v12_mse"]["sweep"]
            for k in SEEDS]
    x = [int(k.split("%")[0]) for k in ORDER]
    stat = lambda k, f: (st.mean(r[k][f] for r in runs), st.stdev(r[k][f] for r in runs))
    soc = [stat(k, "soc_rmse") for k in ORDER]
    soh = [stat(k, "soh_rmse") for k in ORDER]

    fig, ax = plt.subplots(figsize=(7.9, 4.69))
    ax.errorbar(x, [m for m, _ in soc], yerr=[s for _, s in soc], marker="o", color="#1f77b4", linewidth=1.8,
                markersize=7, capsize=3, label="SOC RMSE")
    ax.errorbar(x, [m for m, _ in soh], yerr=[s for _, s in soh], marker="s", color="#ff7f0e", linewidth=1.8,
                markersize=7, capsize=3, label="SOH RMSE")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Perturbation of fixed R0 (%)", fontsize=12)
    ax.set_ylabel("RMSE (%)", fontsize=12)
    ax.set_xticks(x)
    ax.grid(True, color="#d0d0d0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {OUT_PNG}")
    print(f"[OK] {OUT_PDF}")


if __name__ == "__main__":
    main()

"""
Generate the corrected Figure 1 (UAPI-Former + post-hoc calibration workflow)
diagram: parallel SOC/SOH branches from the gates all the way to the
conformal output, IC-inspired proxy shown as an independent input feeding
only the SOH branch, and the S1 refusal flag branching explicitly from the
S1 option rather than being embedded ambiguously in one shared box.

Replaces an earlier version of this figure that drew a single horizontal
chain (SOC NIG head -> SOH NIG head + IC proxy -> one shared conformal
layer), which a reviewer correctly flagged as implying a sequential
architecture the Methods section does not describe.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "figs")

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

GRAY   = ("#F0F0F0", "#616161")
BLUE   = ("#DCEEFB", "#1565C0")   # SOC branch
ORANGE = ("#FDE9D9", "#C2570C")   # SOH branch
GREEN  = ("#E4F4E1", "#2E7D32")   # cross-cutting note (refusal flag)


def box(ax, cx, cy, w, h, colors, lw=1.4):
    face, edge = colors
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06", facecolor=face, edgecolor=edge,
        linewidth=lw, zorder=2,
    ))


def txt(ax, cx, cy, s, size=10.5, bold=False, color="#111"):
    ax.text(cx, cy, s, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=color, zorder=3)


def varrow(ax, x, y0, y1, style="-|>", color="#444", lw=1.3, ls="solid"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, mutation_scale=11), zorder=1)


def harrow(ax, x0, y, x1, style="-|>", color="#444", lw=1.3, ls="solid"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, mutation_scale=11), zorder=1)


def build():
    W, H = 10.6, 10.6
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(W, H))
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

        cx = W / 2
        BW = 4.6
        OFF = 1.95
        cxL, cxR = cx - OFF, cx + OFF   # SOC (left), SOH (right)
        BW2 = 3.2

        Y = dict(bms=9.9, eite=9.0, proj=8.1, tx=7.2, pool=6.35,
                 gate=5.35, headI=4.35, head=4.35, conf=3.15, out=2.05)
        HS = 0.62

        # ── shared backbone (single column) ──────────────────────────
        box(ax, cx, Y["bms"], BW, HS, GRAY)
        txt(ax, cx, Y["bms"], "BMS signals: V, I, T  (T = 200 timesteps)", size=9.5)
        varrow(ax, cx, Y["bms"] - HS/2, Y["eite"] + HS/2)

        box(ax, cx, Y["eite"], BW, HS, GRAY)
        txt(ax, cx, Y["eite"], "EITE features: Vnorm, I, Tnorm, VOCV, zOCV, Qcum", size=9)
        varrow(ax, cx, Y["eite"] - HS/2, Y["proj"] + HS/2)

        box(ax, cx, Y["proj"], BW, HS, GRAY)
        txt(ax, cx, Y["proj"], "Projection + window-conditioned chemistry token", size=9.5)
        varrow(ax, cx, Y["proj"] - HS/2, Y["tx"] + HS/2)

        box(ax, cx, Y["tx"], BW, HS, GRAY)
        txt(ax, cx, Y["tx"], "4-layer Transformer encoder", size=10)
        varrow(ax, cx, Y["tx"] - HS/2, Y["pool"] + HS/2)

        box(ax, cx, Y["pool"], BW, HS, GRAY, lw=1.6)
        txt(ax, cx, Y["pool"], "Physics-residual pooling (PRAP)", size=10, bold=True)

        # split into two parallel columns
        y_split = (Y["pool"] - HS/2 + Y["gate"] + HS/2) / 2
        ax.plot([cx, cx], [Y["pool"] - HS/2, y_split], color="#555", lw=1.2, zorder=1)
        ax.plot([cx, cxL], [y_split, y_split], color="#555", lw=1.2, zorder=1)
        ax.plot([cx, cxR], [y_split, y_split], color="#555", lw=1.2, zorder=1)
        varrow(ax, cxL, y_split, Y["gate"] + HS/2)
        varrow(ax, cxR, y_split, Y["gate"] + HS/2)

        # ── DTAG gates (parallel) ─────────────────────────────────────
        box(ax, cxL, Y["gate"], BW2, HS, BLUE, lw=1.6)
        txt(ax, cxL, Y["gate"], "SOC gate (DTAG)", bold=True, color="#0D47A1")
        box(ax, cxR, Y["gate"], BW2, HS, ORANGE, lw=1.6)
        txt(ax, cxR, Y["gate"], "SOH gate (DTAG)", bold=True, color="#8A3B00")

        # CTBA: residual cross-task projection between the two gated vectors (R2.7: not attention, not an interaction claim)
        gap_l = cxL + BW2/2 + 0.05
        gap_r = cxR - BW2/2 - 0.05
        ax.annotate("", xy=(gap_r, Y["gate"]), xytext=(gap_l, Y["gate"]),
                    arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6,
                                    mutation_scale=11), zorder=1)
        ax.text(cx, Y["gate"] + 0.42, "CTBA: cross-task residual adapter",
                ha="center", va="center", fontsize=8.3,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="#AAAAAA", lw=0.8), zorder=4)

        varrow(ax, cxL, Y["gate"] - HS/2, Y["head"] + HS/2)
        varrow(ax, cxR, Y["gate"] - HS/2, Y["head"] + HS/2)

        # ── IC-inspired proxy: independent input feeding ONLY the SOH branch ──
        icx, icy = cxR + BW2/2 + 0.9, Y["head"]
        box(ax, icx, icy, 1.55, 0.5, GREEN, lw=1.3)
        txt(ax, icx, icy, "IC-inspired\nproxy", size=8.6, color="#1B5E20")
        ax.annotate("", xy=(cxR + BW2/2 + 0.02, Y["head"]),
                    xytext=(icx - 0.775, icy),
                    arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.2,
                                    mutation_scale=10), zorder=1)

        # ── NIG heads (parallel) ────────────────────────────────────────
        box(ax, cxL, Y["head"], BW2, HS, BLUE)
        txt(ax, cxL, Y["head"], "SOC NIG head\n(point estimate + uncertainty)", size=9, color="#0D47A1")
        box(ax, cxR, Y["head"], BW2, HS, ORANGE)
        txt(ax, cxR, Y["head"], "SOH NIG head\n(point estimate + uncertainty)", size=9, color="#8A3B00")

        varrow(ax, cxL, Y["head"] - HS/2, Y["conf"] + HS/2)
        varrow(ax, cxR, Y["head"] - HS/2, Y["conf"] + HS/2)

        # ── conformal layers (parallel, independent per-task choice) ────
        box(ax, cxL, Y["conf"], BW2, HS, BLUE, lw=1.6)
        txt(ax, cxL, Y["conf"], "SOC conformal layer\n(S0 / S1 / S2, post hoc)", size=8.8, color="#0D47A1")
        box(ax, cxR, Y["conf"], BW2, HS, ORANGE, lw=1.6)
        txt(ax, cxR, Y["conf"], "SOH conformal layer\n(S0 / S1 / S2, post hoc)", size=8.8, color="#8A3B00")

        # refusal flag branches explicitly from S1, shown as a shared
        # cross-cutting note just above the conformal-layer row (mirroring
        # how the CTBA note sits just above the gate row)
        rfy = Y["conf"] + HS/2 + 0.30
        ax.text(cx, rfy, "S1 refusal flag: insufficient calibration support",
                ha="center", va="center", fontsize=7.8, color="#1B5E20",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#E4F4E1",
                          edgecolor="#2E7D32", lw=0.9), zorder=4)
        ax.annotate("", xy=(cxL, Y["conf"] + HS/2 + 0.04), xytext=(cx - 0.85, rfy - 0.14),
                    arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.0,
                                    linestyle="dashed", mutation_scale=8), zorder=1)
        ax.annotate("", xy=(cxR, Y["conf"] + HS/2 + 0.04), xytext=(cx + 0.85, rfy - 0.14),
                    arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.0,
                                    linestyle="dashed", mutation_scale=8), zorder=1)

        varrow(ax, cxL, Y["conf"] - HS/2, Y["out"] + HS/2)
        varrow(ax, cxR, Y["conf"] - HS/2, Y["out"] + HS/2)

        # ── outputs (parallel) ────────────────────────────────────────
        box(ax, cxL, Y["out"], BW2, HS, BLUE, lw=1.6)
        txt(ax, cxL, Y["out"], "SOC estimate + PI", bold=True, color="#0D47A1")
        box(ax, cxR, Y["out"], BW2, HS, ORANGE, lw=1.6)
        txt(ax, cxR, Y["out"], "SOH estimate + PI", bold=True, color="#8A3B00")

        pdf_path = os.path.join(FIGS, "workflow_figure1.pdf")
        png_path = os.path.join(FIGS, "workflow_figure1.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {pdf_path}")
        print(f"[OK] {png_path}")


if __name__ == "__main__":
    build()

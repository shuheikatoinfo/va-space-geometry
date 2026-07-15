"""Graphical abstract (money figure): one operating point, two harms.

Left: schematic of the domain — professional voices crowd the embedding space
relative to a general-public control. Right: the threshold trade-off — a clone
of an enrolled actor slips below the real-calibrated threshold (missed), while
a clone of a non-enrolled person lands on an innocent enrolled actor above it
(wrongful accusation). Headline numbers match the paper (generic encoder).

Intended uses: IEEE Access graphical abstract, arXiv/social card.
Usage: python -m src.fig_graphical_abstract
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INK = "#1a1a24"; SEC = "#4a4a55"; MUT = "#8a8a94"
RED = "#e45756"; BLUE = "#3b76c9"; GRAY_DOT = "#9aa3ad"
SURFACE = "#ffffff"


def left_panel(ax):
    rng = np.random.default_rng(7)
    n = 42
    pub = rng.normal([-1.45, 0.0], 0.52, (n, 2))
    va = rng.normal([1.45, 0.0], 0.24, (n, 2))
    ax.scatter(*pub.T, s=26, c=GRAY_DOT, lw=0, alpha=0.9)
    ax.scatter(*va.T, s=26, c=SEC, lw=0, alpha=0.95)
    ax.text(-1.45, -1.08, "general public\n(control)", ha="center", va="top",
            fontsize=11, color=SEC)
    ax.text(1.45, -1.08, "professional\nvoice actors", ha="center", va="top",
            fontsize=11, color=INK, fontweight="bold")
    ax.set_title("Trained voices crowd the embedding space",
                 fontsize=13, color=INK, fontweight="bold", pad=10)
    ax.text(0, -1.95,
            "misidentification floor survives calibration and\n"
            "re-ranking: 2.6% (best ensemble; 13.0% session-\n"
            "disjoint) — several-fold above matched controls",
            ha="center", va="top", fontsize=10.5, color=SEC, linespacing=1.35)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-3.4, 1.4)
    ax.set_aspect("equal"); ax.axis("off")


def right_panel(ax):
    tau = 0.54
    ax.axhline(tau, ls=(0, (5, 4)), lw=2, color=SEC, zorder=1)
    ax.text(0.985, tau - 0.025, "threshold $\\tau$\n(calibrated on real voices)",
            ha="right", va="top", fontsize=10.5, color=SEC, linespacing=1.25)

    # --- scenario 1: enrolled actor's clone slips below tau ---
    x1 = 0.24
    ax.scatter([x1], [0.80], s=120, facecolors="none", edgecolors=MUT,
               lw=2, zorder=3)
    ax.text(x1 + 0.035, 0.80, "where a real match scores", ha="left",
            va="center", fontsize=10, color=MUT)
    ax.annotate("", xy=(x1, 0.40), xytext=(x1, 0.755),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.text(x1 + 0.035, 0.60, "synthetic-vs-real\nshift", ha="left",
            va="center", fontsize=10, color=SEC, linespacing=1.2)
    ax.scatter([x1], [0.365], s=130, c=BLUE, lw=0, zorder=3)
    ax.text(x1, 0.29, "clone of enrolled actor A\nfalls below $\\tau$",
            ha="center", va="top", fontsize=11, color=INK, linespacing=1.3)
    ax.text(x1, 0.155, "MISSED: 32% of Seed-VC clones",
            ha="center", va="top", fontsize=11.5, color=INK, fontweight="bold")

    # --- scenario 2: non-enrolled clone lands on innocent actor B ---
    x2 = 0.72
    ax.scatter([x2], [0.70], s=130, c=RED, lw=0, zorder=3)
    ax.scatter([x2 + 0.055], [0.735], s=120, facecolors="none",
               edgecolors=SEC, lw=2, zorder=3)
    ax.text(x2 + 0.085, 0.735, "innocent\nactor B", ha="left", va="center",
            fontsize=10, color=SEC, linespacing=1.2)
    ax.annotate("", xy=(x2, 0.665), xytext=(x2, 0.30),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(x2, 0.29, "clone of a non-enrolled person\nlands on actor B, above $\\tau$",
            ha="center", va="top", fontsize=11, color=INK, linespacing=1.3)
    ax.text(x2, 0.155, "WRONGFUL ACCUSATION:\n$\\approx$half of non-enrolled clones",
            ha="center", va="top", fontsize=11.5, color=INK,
            fontweight="bold", linespacing=1.25)

    # --- trade-off arrow in the gap between the two scenarios ---
    xm = 0.485
    ax.annotate("", xy=(xm, tau + 0.14), xytext=(xm, tau - 0.14),
                arrowprops=dict(arrowstyle="<|-|>", color=MUT, lw=1.6))
    ax.text(xm, tau + 0.165, "raise $\\tau$: more misses", ha="center",
            va="bottom", fontsize=9.5, color=MUT)
    ax.text(xm, tau - 0.165, "lower $\\tau$: more accusations", ha="center",
            va="top", fontsize=9.5, color=MUT)

    ax.set_title("One operating point, two harms (generic encoder)",
                 fontsize=13, color=INK, fontweight="bold", pad=10)
    ax.set_ylabel("similarity of clone to its top-ranked enrolled actor",
                  fontsize=10.5, color=SEC)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUT)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["standalone", "overview"],
                    default="standalone",
                    help="standalone = IEEE Access graphical abstract / social "
                         "card (with title); overview = compact in-paper Fig. 1 "
                         "(no title, the caption carries the message)")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "text.color": INK})

    if args.variant == "standalone":
        fig = plt.figure(figsize=(13.2, 6.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55],
                              left=0.045, right=0.975, top=0.80, bottom=0.17,
                              wspace=0.16)
        left_panel(fig.add_subplot(gs[0]))
        right_panel(fig.add_subplot(gs[1]))
        fig.suptitle("Voice-clone attribution by similarity threshold fails both ways\n"
                     "in the dense voice space of professional voice actors",
                     fontsize=17, fontweight="bold", color=INK, y=0.975,
                     linespacing=1.25)
        fig.text(0.5, 0.03,
                 "No single operating point avoids both harms — the limit is the embedding geometry, not the threshold.\n"
                 "A domain-matched (voice-actor-trained) encoder cuts wrongful accusations to 14–18%, but the floor remains.",
                 ha="center", va="bottom", fontsize=11, color=SEC, linespacing=1.45)
        out = Path("output/fig_graphical_abstract.png")
    else:
        # Compact in-paper Fig. 1: no suptitle/footer (the LaTeX caption carries
        # the message), a wider/shorter aspect to spend as little column height
        # as possible.
        fig = plt.figure(figsize=(13.2, 4.15))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6],
                              left=0.045, right=0.985, top=0.90, bottom=0.045,
                              wspace=0.14)
        left_panel(fig.add_subplot(gs[0]))
        right_panel(fig.add_subplot(gs[1]))
        out = Path("output/fig_overview.png")

    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

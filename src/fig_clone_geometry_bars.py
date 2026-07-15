"""Alternative to the clone-geometry UMAP (src/clone_geometry.py --umap): a
quantitative bar figure that states the same message directly from the cached
metrics in output/analysis/clone_geometry.json, no re-embedding needed.

Message: clones cluster by TARGET SPEAKER, not by SYNTHESIZER.
  - left  : silhouette coefficient of the clone embeddings, by target vs by method
  - right : nearest-neighbour enrichment over chance, same-target vs same-method (log scale)

Usage: python -m src.fig_clone_geometry_bars [--models ecapa animeva]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AN = Path("output/analysis")

DISP_NAMES = {
    "ecapa": "ECAPA-TDNN", "animeva": "animeva", "campp": "CAM++",
    "redimnet": "ReDimNet-b2", "xvector": "x-vector", "wavlm": "WavLM-base-plus-sv",
    "jhubert": "JP-HuBERT", "jxvector": "jxvector", "ens_sv4": "SV-4", "ens_all6": "all-6",
}
TARGET_C = "#4c78a8"   # target-speaker series
METHOD_C = "#e45756"   # synthesizer series


def disp(k: str) -> str:
    return DISP_NAMES.get(str(k), str(k))


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantitative bar version of the clone-geometry figure.")
    ap.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    args = ap.parse_args()

    d = json.load(open(AN / "clone_geometry.json"))
    models = [m for m in args.models if m in d]
    if not models:
        raise SystemExit("No requested models found in clone_geometry.json.")

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 11, "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(models)); w = 0.36

    # --- left: silhouette by target vs by synthesizer ---
    sil_t = [d[m]["silhouette_by_target"] for m in models]
    sil_m = [d[m]["silhouette_by_method"] for m in models]
    ax[0].bar(x - w / 2, sil_t, w, color=TARGET_C, label="by target speaker")
    ax[0].bar(x + w / 2, sil_m, w, color=METHOD_C, label="by synthesizer")
    ax[0].set_xticks(x); ax[0].set_xticklabels([disp(m) for m in models])
    ax[0].set_ylabel("silhouette coefficient")
    ax[0].set_title("Clones cluster by target speaker,\nnot by synthesizer", fontsize=12, fontweight="bold")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
    for xi, (a, b) in enumerate(zip(sil_t, sil_m)):
        ax[0].text(xi - w / 2, a + 0.005, f"{a:.2f}", ha="center", va="bottom", fontsize=10)
        ax[0].text(xi + w / 2, b + 0.005, f"{b:.2f}", ha="center", va="bottom", fontsize=10)

    # --- right: nearest-neighbour enrichment over chance (log scale) ---
    enr_t = [d[m]["nn_same_target"] / d[m]["nn_chance_target"] for m in models]
    enr_m = [d[m]["nn_same_method"] / d[m]["nn_chance_method"] for m in models]
    ax[1].bar(x - w / 2, enr_t, w, color=TARGET_C, label="same target speaker")
    ax[1].bar(x + w / 2, enr_m, w, color=METHOD_C, label="same synthesizer")
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([disp(m) for m in models])
    ax[1].set_ylabel("NN enrichment over chance ($\\times$)")
    ax[1].set_title("A clone's nearest neighbour shares its target\nfar more than its synthesizer", fontsize=12, fontweight="bold")
    ax[1].axhline(1, color="k", lw=0.8, ls=":")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
    for xi, (a, b) in enumerate(zip(enr_t, enr_m)):
        ax[1].text(xi - w / 2, a * 1.05, f"{a:.0f}$\\times$", ha="center", va="bottom", fontsize=10)
        ax[1].text(xi + w / 2, b * 1.05, f"{b:.0f}$\\times$", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out = Path("output/fig_clone_geometry_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    for m in models:
        print(f"  {disp(m):12s} silhouette target={d[m]['silhouette_by_target']:.3f} method={d[m]['silhouette_by_method']:.3f}; "
              f"NN enrichment target={enr_t[models.index(m)]:.0f}x method={enr_m[models.index(m)]:.0f}x")


if __name__ == "__main__":
    main()

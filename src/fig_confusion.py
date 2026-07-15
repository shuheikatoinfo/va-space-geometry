"""Figure: confusion-graph structure, honest version.

Left panel visualizes the in-degree RETRACTION — the observed speaker in-degree
distribution overlaps the size-proportional null (target ∝ segment count) but
is far above the uniform null; i.e. the apparent hubness is a segment-count
artifact. Right panel shows the two structural properties that DO survive
(modularity and reciprocity) as observed value vs null band.

Reads output/analysis/confusion_null_{model}.json and reconstructs the
in-degree draws from src.confusion_null.

Usage: python -m src.fig_confusion --model ecapa
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.confusion_null import observed_edges, null_draws


def indeg_dist(src, tgt, nspk, weights, rng, reps=30):
    """Collect pooled in-degree values across `reps` null graphs (for a smooth
    histogram of the null in-degree distribution)."""
    vals = []
    p = None if weights is None else weights / weights.sum()
    for _ in range(reps):
        tn = rng.choice(nspk, size=len(src), p=p)
        bad = tn == src
        while bad.any():
            tn[bad] = rng.choice(nspk, size=int(bad.sum()), p=p); bad = tn == src
        vals.append(np.bincount(tn, minlength=nspk))
    return np.concatenate(vals)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ecapa")
    args = ap.parse_args()
    d = json.load(open(f"output/analysis/confusion_null_{args.model}.json"))
    src, tgt, nspk, seg_counts = observed_edges(args.model)
    obs_indeg = np.bincount(tgt, minlength=nspk)
    rng = np.random.default_rng(0)
    null_sizeprop = indeg_dist(src, tgt, nspk, seg_counts.astype(float), rng)
    null_uniform = indeg_dist(src, tgt, nspk, None, rng)

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.8))
    # --- left: in-degree retraction ---
    hi = int(np.percentile(np.concatenate([obs_indeg, null_sizeprop]), 99.5))
    bins = np.linspace(0, max(hi, 5), 40)
    ax[0].hist(null_uniform, bins=bins, density=True, alpha=.45, color="#bbbbbb",
               label=f"uniform null (skew {d['indeg_skew_null_mean_uniform']:.2f})")
    ax[0].hist(null_sizeprop, bins=bins, density=True, alpha=.55, color="#4c78a8",
               label=f"size-proportional null (skew {d['indeg_skew_null_mean']:.2f})")
    ax[0].hist(obs_indeg, bins=bins, density=True, histtype="step", lw=2.2, color="#e45756",
               label=f"observed (skew {d['indeg_skew_obs']:.2f})")
    ax[0].set_xlabel("speaker in-degree\n(# queries taking speaker as nearest impostor)")
    ax[0].set_ylabel("density")
    p_skew = d["indeg_skew_p_emp"]
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=True)

    # --- right: surviving structure ---
    labels = ["modularity", "reciprocity\n(1 - one-directional)"]
    obs = [d["modularity_obs"], 1 - d["asymmetry_obs"]]
    nmean = [d["modularity_null_mean"], 1 - d["asymmetry_null_mean"]]
    nstd = [d.get("modularity_null_std", 0), d.get("asymmetry_null_std", 0)]
    pemp = [d["modularity_p_emp"], d["asymmetry_p_emp"]]
    x = np.arange(len(labels))
    ax[1].bar(x - .18, obs, .36, color="#e45756", label="observed")
    ax[1].bar(x + .18, nmean, .36, yerr=np.array(nstd) * 2, color="#4c78a8", alpha=.7,
              label="null (mean±2σ)", capsize=4)
    for i, p in enumerate(pemp):
        ax[1].text(i, max(obs[i], nmean[i]) + .03, f"$p={p:.2g}$", ha="center", fontsize=11)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylim(0, 1.0); ax[1].set_ylabel("value")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)

    fig.tight_layout()
    out = Path("output/fig_confusion_indegree.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

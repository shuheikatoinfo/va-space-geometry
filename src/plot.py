"""Phase 5: visualization and metrics summary.

Produces, in output/:
  - margin_distribution.png : histogram of identification (verification) margins
        per model, with the near-zero "critical" region shaded. Visual proof of
        how thin the margins are under a single fixed cosine threshold.
  - speaker_space_map.png   : 2-D (UMAP, t-SNE fallback) projection per model,
        colored by verification margin so thin-margin critical points and dense
        clusters are visible.
  - metrics_summary.json    : combined numerical summary across all models.

Usage:
   python -m src.plot [--models ...] [--reducer umap|tsne]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path("output/analysis")
EMB_DIR = Path("output/embeddings")
OUT_DIR = Path("output")
CRITICAL_BAND = 0.02

DISP_NAMES = {
    "ecapa": "ECAPA-TDNN", "animeva": "animeva", "campp": "CAM++",
    "redimnet": "ReDimNet-b2", "xvector": "x-vector", "wavlm": "WavLM-base-plus-sv",
    "jhubert": "JP-HuBERT", "hubert": "JP-HuBERT", "jxvector": "jxvector",
    "ens_sv4": "SV-4", "ens_all6": "all-6", "seedvc": "Seed-VC",
    "irodori": "Irodori-TTS", "gptsovits": "GPT-SoVITS", "gptsovits_v1": "GPT-SoVITS v1",
    "gptsovits_v2": "GPT-SoVITS v2", "gptsovits_v2ProPlus": "GPT-SoVITS v2ProPlus",
    "gptsovits_v3": "GPT-SoVITS v3", "gptsovits_v4": "GPT-SoVITS v4", "real": "real",
}


def disp(k):
    return DISP_NAMES.get(str(k), str(k))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def reduce_2d(emb: np.ndarray, reducer: str, seed: int) -> np.ndarray:
    xn = l2_normalize(emb)
    if reducer == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(xn)
        except Exception as exc:  # noqa: BLE001
            print(f"  UMAP unavailable ({exc}); falling back to t-SNE.")
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, (len(xn) - 1) // 3))
    return TSNE(n_components=2, metric="cosine", init="random",
                perplexity=perplexity, random_state=seed).fit_transform(xn)


def plot_margins(models: list[str]) -> None:
    cols = min(3, len(models))
    rows = (len(models) + cols - 1) // cols
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.5 * rows), squeeze=False)
    n = len(models)
    for i, (ax, name) in enumerate(zip(axes.flat, models)):
        d = np.load(ANALYSIS_DIR / f"{name}.npz", allow_pickle=True)
        vm = d["verification_margin"][d["has_same"]]
        ax.hist(vm, bins=80, color="#3b6ea5", alpha=0.85)
        ax.axvspan(-CRITICAL_BAND, CRITICAL_BAND, color="red", alpha=0.15,
                   label=f"|margin|<{CRITICAL_BAND}")
        ax.axvline(0, color="k", lw=1)
        # cap the number of x-ticks so narrow-range panels (e.g. x-vector, ±0.05)
        # do not collide their tick labels
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))
        # short in-panel encoder label (formal name; no descriptive title)
        ax.annotate(disp(name), xy=(0.96, 0.93), xycoords="axes fraction",
                    ha="right", va="top", fontsize=11, fontweight="bold")
        if i >= n - cols:
            ax.set_xlabel("identification margin")
        if i % cols == 0:
            ax.set_ylabel("count")
    # single shared legend placed in a genuinely empty grid cell (never over data)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    empties = list(axes.flat[len(models):])
    for ax in empties:
        ax.axis("off")
    if empties:
        empties[0].legend(handles, labels, loc="center", frameon=True, fontsize=11)
    else:
        axes.flat[0].legend(handles, labels, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_margin_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_margin_distribution.png'}")


def plot_space_maps(models: list[str], reducer: str, seed: int) -> None:
    cols = min(3, len(models))
    rows = (len(models) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)
    for ax, name in zip(axes.flat, models):
        emb = np.load(EMB_DIR / f"{name}.npz", allow_pickle=True)["emb"].astype(np.float32)
        an = np.load(ANALYSIS_DIR / f"{name}.npz", allow_pickle=True)
        vm = an["verification_margin"].astype(np.float32)
        xy = reduce_2d(emb, reducer, seed)
        # Color by verification margin; clip so the critical region is salient.
        c = np.clip(vm, -0.1, 0.3)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="coolwarm_r", s=6, alpha=0.8)
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8, label="verification margin")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Speaker-embedding space ({reducer}); red = thin-margin critical points")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "speaker_space_map.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'speaker_space_map.png'}")


def plot_source_maps(models: list[str], reducer: str, seed: int) -> None:
    """2-D projection colored by recording source, to reveal channel clusters.

    If points cluster by recording source (agency studio / platform) rather than
    by speaker, thin margins may be a microphone/environment artifact. This panel
    makes that visually checkable next to speaker_space_map.png.
    """
    cols = min(3, len(models))
    rows = (len(models) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)
    for ax, name in zip(axes.flat, models):
        emb = np.load(EMB_DIR / f"{name}.npz", allow_pickle=True)["emb"].astype(np.float32)
        an = np.load(ANALYSIS_DIR / f"{name}.npz", allow_pickle=True)
        src = an["recording_source"] if "recording_source" in an else np.array(["?"] * len(emb))
        xy = reduce_2d(emb, reducer, seed)
        # Color by the (top-K) most frequent sources; rest grey.
        labels, counts = np.unique(src, return_counts=True)
        top = set(labels[np.argsort(-counts)][:12])
        codes = np.array([list(top).index(s) if s in top else -1 for s in src])
        ax.scatter(xy[codes == -1, 0], xy[codes == -1, 1], c="lightgrey", s=5, alpha=0.5)
        sc = ax.scatter(xy[codes >= 0, 0], xy[codes >= 0, 1],
                        c=codes[codes >= 0], cmap="tab20", s=6, alpha=0.85)
        ax.set_title(f"{name} (by recording source)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Recording-source coloring ({reducer}) — channel-confound check")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "source_space_map.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'source_space_map.png'}")


def write_summary(models: list[str]) -> None:
    combined = {"models": {}, "comparison": []}
    for name in models:
        with (ANALYSIS_DIR / f"{name}.json").open(encoding="utf-8") as fh:
            s = json.load(fh)
        combined["models"][name] = s
        combined["comparison"].append({
            "model": name,
            "embedding_dim": s["embedding_dim"],
            "rank1_accuracy": s["phase3_margin"]["rank1_accuracy"],
            "misidentified_fraction": s["phase3_margin"]["misidentified_fraction"],
            "critical_<0.02": s["phase3_margin"]["critical_fraction_abs_margin"].get("<0.02"),
            "hubness_skewness": s["phase4_geometry"]["hubness_skewness"],
            "antihub_fraction": s["phase4_geometry"]["antihub_fraction"],
            "same_source_impostor_enrichment": s.get("channel_confound", {}).get("same_source_enrichment"),
        })
    with (OUT_DIR / "metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2)
    print(f"  wrote {OUT_DIR / 'metrics_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 plots and summary.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--reducer", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.models is None:
        # per-encoder analysis outputs have BOTH a {name}.json summary and a
        # {name}.npz; other analysis JSONs (clone_score_*, *_falsematch,
        # contamination_audit, ...) have no matching npz and must be skipped.
        args.models = sorted(p.stem for p in ANALYSIS_DIR.glob("*.npz"))
    if not args.models:
        raise SystemExit("No analysis results found. Run Phase 2-4 (src.analyze) first.")

    plot_margins(args.models)
    plot_space_maps(args.models, args.reducer, args.seed)
    plot_source_maps(args.models, args.reducer, args.seed)
    write_summary(args.models)


if __name__ == "__main__":
    main()

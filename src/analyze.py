"""Phases 2-4: cosine similarity, identification margins, geometry & hubness.

For each model's embeddings (output/embeddings/<model>.npz) this computes:

Phase 2 - cosine similarity (implicitly, via L2-normalized dot products; the full
          N x N matrix is processed in GPU chunks, never fully materialized).

Phase 3 - identification margin (the central analysis):
   For every query segment q:
     same_best = max cosine sim to *other* segments of the SAME speaker
     diff_best = max cosine sim to segments of a DIFFERENT speaker
   Two margin definitions are reported:
     - verification margin  = same_best - diff_best
         (>0 means the right speaker is closer than any impostor; near 0 means a
          tiny perturbation flips identity -> over-block / false accept risk)
     - operational top1-top2 = s_top1 - s_top2 over all other segments
   Critical queries are those with |verification margin| below small thresholds;
   misidentified queries have verification margin < 0. We also tally the
   speaker pairs that compete in the small-margin region.

Phase 4 - geometry:
   - local density: per-point mean cosine distance to its k nearest neighbors;
     skewness of that distribution
   - nearest-neighbor distance distribution skewness
   - hubness: k-occurrence N_k(x) (how often x is in others' k-NN); the skewness
     of N_k is the hubness score (Radovanovic et al.). We report hub / antihub
     counts and a permutation-based significance check on the skewness.

Outputs:
   output/analysis/<model>.json          -- summary metrics
   output/analysis/<model>.npz           -- arrays for plotting

Usage:
   python -m src.analyze [--models ...] [--k 10] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats

EMB_DIR = Path("output/embeddings")
OUT_DIR = Path("output/analysis")
CRITICAL_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def analyze_model(name: str, k: int, device: str, seed: int) -> dict:
    data = np.load(EMB_DIR / f"{name}.npz", allow_pickle=True)
    emb = data["emb"].astype(np.float32)
    speaker_id = data["speaker_id"]
    segment_id = data["segment_id"]
    recording_source = data["recording_source"] if "recording_source" in data else np.array(["unknown"] * len(emb))
    n, d = emb.shape
    if n < 3:
        raise SystemExit(f"{name}: not enough segments ({n}) to analyze.")

    Xn = torch.from_numpy(l2_normalize(emb)).to(device)
    spk = np.asarray(speaker_id)
    # Integer speaker codes for fast same/diff masks on GPU.
    _, spk_codes = np.unique(spk, return_inverse=True)
    spk_t = torch.from_numpy(spk_codes).to(device)

    same_best = np.full(n, -2.0, dtype=np.float32)
    diff_best = np.full(n, -2.0, dtype=np.float32)
    diff_best_idx = np.full(n, -1, dtype=np.int64)
    op_top1 = np.full(n, -2.0, dtype=np.float32)
    op_top2 = np.full(n, -2.0, dtype=np.float32)
    knn_density = np.zeros(n, dtype=np.float32)       # mean cosine distance to k-NN
    nn_distance = np.zeros(n, dtype=np.float32)       # cosine distance to nearest
    occurrence = np.zeros(n, dtype=np.int64)          # k-occurrence (hubness)

    chunk = 2048
    neg = torch.tensor(-2.0, device=device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = Xn[start:end] @ Xn.T                   # (c, n) cosine sims
        rows = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), rows] = neg  # mask self

        same_mask = spk_t.unsqueeze(0) == spk_t[start:end].unsqueeze(1)
        same_sims = torch.where(same_mask, sims, neg)
        diff_sims = torch.where(same_mask, neg, sims)

        same_best[start:end] = same_sims.max(dim=1).values.cpu().numpy()
        db_val, db_idx = diff_sims.max(dim=1)
        diff_best[start:end] = db_val.cpu().numpy()
        diff_best_idx[start:end] = db_idx.cpu().numpy()

        # operational top-2 over all others
        top2 = sims.topk(2, dim=1).values
        op_top1[start:end] = top2[:, 0].cpu().numpy()
        op_top2[start:end] = top2[:, 1].cpu().numpy()

        # k-NN for density + hubness
        topk = sims.topk(k, dim=1)
        knn_sims = topk.values
        knn_idx = topk.indices
        knn_density[start:end] = (1.0 - knn_sims).mean(dim=1).cpu().numpy()
        nn_distance[start:end] = (1.0 - knn_sims[:, 0]).cpu().numpy()
        flat = knn_idx.reshape(-1).cpu().numpy()
        np.add.at(occurrence, flat, 1)

    # --- Phase 3 metrics ---
    has_same = same_best > -1.5  # speaker has >=1 other segment
    ver_margin = same_best - diff_best
    vm = ver_margin[has_same]
    op_margin = op_top1 - op_top2

    rank1_acc = float(np.mean(same_best[has_same] > diff_best[has_same])) if has_same.any() else float("nan")
    misident_frac = float(np.mean(vm < 0)) if vm.size else float("nan")
    critical_frac = {f"<{t}": float(np.mean(np.abs(vm) < t)) for t in CRITICAL_THRESHOLDS} if vm.size else {}

    # Competing speaker pairs in the small-margin (|margin| < 0.05) region.
    pair_counter: Counter = Counter()
    near = np.where(has_same & (np.abs(ver_margin) < 0.05))[0]
    for i in near:
        j = diff_best_idx[i]
        if j >= 0:
            a, b = sorted([str(spk[i]), str(spk[j])])
            pair_counter[f"{a} | {b}"] += 1
    top_pairs = pair_counter.most_common(30)

    # --- Channel-confound metrics ---
    # Is the nearest impostor disproportionately from the same recording source
    # (agency studio / platform)? If so, thin margins are partly a channel effect,
    # not pure speaker proximity. Compare to the chance level of same-source.
    src = np.asarray(recording_source)
    valid = diff_best_idx >= 0
    comp_src = np.where(valid, src[diff_best_idx.clip(min=0)], "")
    same_src_comp = (comp_src == src) & valid
    # Chance level: per query, fraction of *other* segments sharing its source.
    _, src_codes2, src_counts = np.unique(src, return_inverse=True, return_counts=True)
    chance_same = float(np.mean((src_counts[src_codes2] - 1) / max(n - 1, 1)))
    nearest_same_src = float(np.mean(same_src_comp[valid])) if valid.any() else float("nan")

    crit_mask = has_same & (np.abs(ver_margin) < 0.05)
    crit_same_src = float(np.mean(same_src_comp[crit_mask])) if crit_mask.any() else float("nan")
    # Margins split by whether the nearest impostor is same vs different source.
    vm_same = ver_margin[has_same & same_src_comp]
    vm_diff = ver_margin[has_same & ~same_src_comp & valid]
    channel = {
        "num_recording_sources": int(len(set(src.tolist()))),
        "nearest_impostor_same_source_fraction": nearest_same_src,
        "chance_same_source_fraction": chance_same,
        "same_source_enrichment": (nearest_same_src / chance_same) if chance_same else None,
        "critical_queries_same_source_fraction": crit_same_src,
        "margin_mean_when_impostor_same_source": float(np.mean(vm_same)) if vm_same.size else None,
        "margin_mean_when_impostor_diff_source": float(np.mean(vm_diff)) if vm_diff.size else None,
    }

    # --- Phase 4 metrics ---
    density_skew = float(stats.skew(knn_density))
    nn_skew = float(stats.skew(nn_distance))
    occ = occurrence.astype(np.float64)
    hubness_skew = float(stats.skew(occ))
    mu, sd = occ.mean(), occ.std()
    hub_count = int(np.sum(occ > mu + 2 * sd))
    antihub_count = int(np.sum(occ == 0))

    # Permutation significance for hubness skewness: shuffle which points are
    # neighbors by resampling occurrence counts under a null of uniform k-NN.
    rng = np.random.default_rng(seed)
    total_edges = int(occ.sum())
    null_skews = []
    for _ in range(200):
        null = np.bincount(rng.integers(0, n, size=total_edges), minlength=n).astype(np.float64)
        null_skews.append(stats.skew(null))
    null_skews = np.array(null_skews)
    hubness_p = float(np.mean(null_skews >= hubness_skew))

    summary = {
        "model": name,
        "num_segments": int(n),
        "embedding_dim": int(d),
        "num_speakers": int(len(set(spk.tolist()))),
        "k": k,
        "phase3_margin": {
            "rank1_accuracy": rank1_acc,
            "verification_margin_mean": float(np.mean(vm)) if vm.size else None,
            "verification_margin_median": float(np.median(vm)) if vm.size else None,
            "verification_margin_std": float(np.std(vm)) if vm.size else None,
            "misidentified_fraction": misident_frac,
            "critical_fraction_abs_margin": critical_frac,
            "operational_top1_top2_margin_mean": float(np.mean(op_margin)),
            "top_competing_speaker_pairs": top_pairs,
        },
        "channel_confound": channel,
        "phase4_geometry": {
            "knn_density_skewness": density_skew,
            "nn_distance_skewness": nn_skew,
            "hubness_skewness": hubness_skew,
            "hubness_skewness_pvalue": hubness_p,
            "hub_count": hub_count,
            "antihub_count": antihub_count,
            "antihub_fraction": float(antihub_count / n),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{name}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    np.savez(
        OUT_DIR / f"{name}.npz",
        verification_margin=ver_margin,
        has_same=has_same,
        operational_margin=op_margin,
        knn_density=knn_density,
        occurrence=occurrence,
        same_best=same_best,
        diff_best=diff_best,
        speaker_id=spk,
        segment_id=segment_id,
        recording_source=src,
        nearest_impostor_same_source=same_src_comp,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Margin & geometry analysis of embeddings.")
    parser.add_argument("--models", nargs="+", default=None, help="Default: all found in output/embeddings.")
    parser.add_argument("--k", type=int, default=10, help="k for k-NN density & hubness.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.models is None:
        args.models = sorted(p.stem for p in EMB_DIR.glob("*.npz"))
    if not args.models:
        raise SystemExit("No embeddings found. Run Phase 1 (src.extract) first.")

    for name in args.models:
        print(f"=== {name} ===")
        s = analyze_model(name, args.k, args.device, args.seed)
        m = s["phase3_margin"]
        g = s["phase4_geometry"]
        print(f"  rank1_acc={m['rank1_accuracy']:.4f} "
              f"misident={m['misidentified_fraction']:.4f} "
              f"crit<0.02={m['critical_fraction_abs_margin'].get('<0.02')} "
              f"hubness_skew={g['hubness_skewness']:.3f} (p={g['hubness_skewness_pvalue']:.3f})")


if __name__ == "__main__":
    main()

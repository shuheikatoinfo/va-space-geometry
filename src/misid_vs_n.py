"""TASK 2 (§7 gallery-size bullet): closed-set rank-1 misID as a function of
gallery size N.

The paper notes absolute misID depends on gallery size N, because S_diff is a max
over an N-dependent impostor pool (Appendix A), and defers a "misID-vs-N curve" to
future work. This script computes it, isolating how much of the cross-table misID
differences (eval-half N=497 vs full ~1100 vs controls 100/455) is *pure N*.

Protocol (identical to the main closed-set misID, Appendix A / analyze.py):
  - agency-only subset (freelance/YouTube sources excluded), raw cosine on
    L2-normalized embeddings.
  - For each target N we draw `n_draws` random *speaker* subsets of size N (without
    replacement) from the agency gallery, keeping ALL segments of each drawn speaker
    (segments-per-speaker handling identical to the main misID: the gallery is a
    pool of segments, S_diff is the max over the impostor-segment pool).
  - misID = fraction of queries (with >=1 same-speaker other segment) whose margin
    S_same - S_diff < 0, computed over the N-speaker sub-gallery only.
  - report mean +- std over draws.

Usage:
    python -m src.misid_vs_n --models ecapa animeva ens_sv4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

EMB = Path("output/embeddings")
AN = Path("output/analysis")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GRID = [100, 200, 350, 500, 750, 1000]
N_DRAWS = 10


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def load_agency(model):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    spk = np.asarray(d["speaker_id"])
    src = np.asarray(d["recording_source"]) if "recording_source" in d else np.array(["?"] * len(spk))
    keep = np.array([not str(src[i]).startswith("freelance:") for i in range(len(spk))])
    emb = d["emb"][keep].astype(np.float32)
    spk = spk[keep]
    return l2(emb), spk


def misid_on_subset(Xt_full, codes_full, sel_idx):
    """Closed-set rank-1 misID over the sub-gallery defined by segment indices sel_idx."""
    X = Xt_full[sel_idx]
    codes = codes_full[sel_idx]
    n = len(sel_idx)
    neg = torch.tensor(-9.0, device=DEVICE)
    sb = np.full(n, -9.0, np.float32)
    db = np.full(n, -9.0, np.float32)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        sims = X[st:e] @ X.T
        sims[torch.arange(e - st, device=DEVICE), torch.arange(st, e, device=DEVICE)] = neg
        same = codes.unsqueeze(0) == codes[st:e].unsqueeze(1)
        sb[st:e] = torch.where(same, sims, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same, neg, sims).max(1).values.cpu().numpy()
    has = sb > -8.0
    return float(np.mean(sb[has] < db[has]))


def run(model, rng_seed=0):
    emb, spk = load_agency(model)
    uspk = np.array(sorted(set(spk.tolist())))
    n_all = len(uspk)
    _, codes = np.unique(spk, return_inverse=True)
    Xt_full = torch.from_numpy(emb).to(DEVICE)
    codes_full = torch.from_numpy(codes).to(DEVICE)
    spk_to_seg = {s: np.where(spk == s)[0] for s in uspk}

    grid = [N for N in N_GRID if N < n_all] + [n_all]  # include "all"
    rng = np.random.default_rng(rng_seed)
    rows = {}
    for N in grid:
        draws = 1 if N == n_all else N_DRAWS
        vals, segcounts = [], []
        for _ in range(draws):
            chosen = rng.choice(uspk, size=N, replace=False) if N < n_all else uspk
            sel = np.concatenate([spk_to_seg[s] for s in chosen])
            segcounts.append(len(sel))
            vals.append(misid_on_subset(Xt_full, codes_full, sel))
        vals = np.array(vals)
        rows[N] = {"misid_mean": float(vals.mean()), "misid_std": float(vals.std()),
                   "n_draws": int(draws), "mean_segments": int(np.mean(segcounts))}
        print(f"  {model:9s} N={N:5d}  misID {vals.mean()*100:5.2f} +- {vals.std()*100:4.2f}% "
              f"(draws={draws})")
    return {"model": model, "n_all_speakers": int(n_all), "n_grid": grid, "by_N": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ecapa", "animeva", "ens_sv4"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {"n_draws_per_N": N_DRAWS, "n_grid_requested": N_GRID, "models": {}}
    for m in args.models:
        print(f"\n#### {m} ####")
        out["models"][m] = run(m, args.seed)
    AN.mkdir(parents=True, exist_ok=True)
    p = AN / "misid_vs_n.json"
    json.dump(out, open(p, "w"), indent=2)
    # side-by-side table
    print("\n=== misID vs N (raw cosine, agency gallery) ===")
    allNs = sorted(set(n for m in out["models"].values() for n in m["n_grid"]))
    hdr = "N      " + "".join(f"{m:>16s}" for m in args.models)
    print(hdr)
    for N in allNs:
        cells = []
        for m in args.models:
            r = out["models"][m]["by_N"].get(N)
            cells.append(f"{r['misid_mean']*100:6.2f}±{r['misid_std']*100:4.2f}" if r else " " * 16)
        print(f"{N:<7d}" + "".join(f"{c:>16s}" for c in cells))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

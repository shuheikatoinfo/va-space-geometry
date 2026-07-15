"""Significance of the confusion-graph structure (addresses 'modularity is trivially
positive' / multiple-comparison concerns). Compares observed in-degree skew,
asymmetry and modularity to degree/out-preserving NULL models.

Primary null (size-proportional): each query keeps its source speaker but its
nearest-impostor target is reassigned to a random DIFFERENT speaker sampled
with probability proportional to that speaker's SEGMENT COUNT. Edges point at
nearest impostor SEGMENTS, so a speaker with more enrolled segments has
proportionally more chances to be someone's nearest neighbour even with no
voice-level hubness; a uniform null destroys that size baseline and inflates
the significance. The legacy uniform null is still reported for comparison.
For modularity we additionally compare to configuration-model rewires of the
symmetrized graph.

p-values are EMPIRICAL: p = (1 + #{null >= obs}) / (1 + n_null). They are
bounded below by 1/(n_null+1); z-scores are reported as effect sizes only.

Usage: python -m src.confusion_null --model ecapa --nnull 200 --mod-null 200
"""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

EMB = Path("output/embeddings")


def observed_edges(model):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32); spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    X = emb[keep]; spk = spk[keep]
    X = X - X.mean(0, keepdims=True); X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    Xt = torch.from_numpy(X).cuda(); _, codes = np.unique(spk, return_inverse=True); ct = torch.from_numpy(codes).cuda()
    neg = torch.tensor(-9.0, device="cuda"); src_codes = []; tgt_codes = []
    for st in range(0, len(spk), 2048):
        e = min(st + 2048, len(spk)); s = Xt[st:e] @ Xt.T
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1); s = torch.where(same, neg, s)
        nb = s.argmax(1).cpu().numpy()
        src_codes += list(codes[st:e]); tgt_codes += list(codes[nb])
    seg_counts = np.bincount(codes, minlength=len(set(codes.tolist())))
    return np.array(src_codes), np.array(tgt_codes), len(set(codes.tolist())), seg_counts


def metrics_from_edges(src, tgt, nspk):
    indeg = np.bincount(tgt, minlength=nspk).astype(float)
    pairs = set(zip(src.tolist(), tgt.tolist())); pairs = {(a, b) for a, b in pairs if a != b}
    recip = sum(1 for a, b in pairs if (b, a) in pairs)
    asym = 1 - recip / max(len(pairs), 1)
    return float(stats.skew(indeg)), float(asym)


def modularity_obs_null(src, tgt, nspk, n_null=20, seed=0):
    """Modularity of the symmetrized confusion graph vs degree-preserving
    (double-edge-swap) rewired nulls. Both observed and null graphs are
    UNWEIGHTED: double_edge_swap drops the weight attribute on swapped-in
    edges, so a weighted observed-vs-null comparison would be distorted;
    binarizing both sides keeps the test apples-to-apples."""
    import networkx as nx
    def build(s, t):
        G = nx.Graph(); w = Counter()
        for a, b in zip(s, t):
            if a != b: w[tuple(sorted((int(a), int(b))))] += 1
        for (a, b), c in w.items(): G.add_edge(a, b)
        return G
    G = build(src, tgt)
    comm = list(nx.community.greedy_modularity_communities(G))
    mod_obs = nx.community.modularity(G, comm)
    rng = np.random.default_rng(seed); nulls = []
    for _ in range(n_null):
        Gn = G.copy()
        try:
            nx.double_edge_swap(Gn, nswap=2 * Gn.number_of_edges(), max_tries=20 * Gn.number_of_edges(),
                                seed=int(rng.integers(2**31)))
        except Exception:
            pass
        cn = list(nx.community.greedy_modularity_communities(Gn))
        nulls.append(nx.community.modularity(Gn, cn))
    nulls = np.array(nulls)
    z = (mod_obs - nulls.mean()) / (nulls.std() + 1e-9)
    return float(mod_obs), float(nulls.mean()), float(nulls.std()), float(z), nulls


def null_draws(src, tgt, nspk, n_null, weights, rng):
    """Reassign each edge's target to a random different speaker, sampled with
    the given per-speaker probability weights (None = uniform)."""
    sk, ay = [], []
    p = None if weights is None else weights / weights.sum()
    for _ in range(n_null):
        tn = rng.choice(nspk, size=len(src), p=p)
        bad = tn == src
        while bad.any():
            tn[bad] = rng.choice(nspk, size=int(bad.sum()), p=p)
            bad = tn == src
        a, b = metrics_from_edges(src, tn, nspk); sk.append(a); ay.append(b)
    return np.array(sk), np.array(ay)


def p_emp(nulls, obs):
    return float((1 + np.sum(np.asarray(nulls) >= obs)) / (1 + len(nulls)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ecapa")
    ap.add_argument("--nnull", type=int, default=200); ap.add_argument("--mod-null", type=int, default=200)
    args = ap.parse_args()
    src, tgt, nspk, seg_counts = observed_edges(args.model)
    skew_o, asym_o = metrics_from_edges(src, tgt, nspk)
    rng = np.random.default_rng(0)
    # primary null: target sampled proportional to speaker segment count
    sk_p, ay_p = null_draws(src, tgt, nspk, args.nnull, seg_counts.astype(float), rng)
    # legacy null: uniform target (destroys the segment-count baseline; kept for comparison)
    sk_u, ay_u = null_draws(src, tgt, nspk, args.nnull, None, rng)
    mod_o, mod_n, mod_s, mod_z, mod_nulls = modularity_obs_null(src, tgt, nspk, n_null=args.mod_null)
    out = {"model": args.model, "n_speakers": nspk, "n_null": args.nnull, "n_null_modularity": args.mod_null,
           "indeg_skew_obs": skew_o,
           "indeg_skew_null_mean": float(sk_p.mean()), "indeg_skew_null_std": float(sk_p.std()),
           "indeg_skew_z": float((skew_o - sk_p.mean()) / (sk_p.std() + 1e-9)),
           "indeg_skew_p_emp": p_emp(sk_p, skew_o),
           "indeg_skew_null_mean_uniform": float(sk_u.mean()),
           "indeg_skew_z_uniform": float((skew_o - sk_u.mean()) / (sk_u.std() + 1e-9)),
           "asymmetry_obs": asym_o,
           "asymmetry_null_mean": float(ay_p.mean()), "asymmetry_null_std": float(ay_p.std()),
           "asymmetry_z": float((asym_o - ay_p.mean()) / (ay_p.std() + 1e-9)),
           "asymmetry_p_emp": p_emp(-np.array(ay_p), -asym_o),  # obs asym LOWER than null = more mutual
           "asymmetry_null_mean_uniform": float(ay_u.mean()),
           "asymmetry_z_uniform": float((asym_o - ay_u.mean()) / (ay_u.std() + 1e-9)),
           "modularity_obs": mod_o, "modularity_null_mean": mod_n, "modularity_null_std": mod_s,
           "modularity_z": mod_z, "modularity_p_emp": p_emp(mod_nulls, mod_o)}
    print(f"#### {args.model} confusion-graph vs null ({nspk} speakers) ####")
    print(f"  in-degree skew:  obs={skew_o:.2f}  size-prop null={sk_p.mean():.2f}±{sk_p.std():.2f}  z={out['indeg_skew_z']:.1f}  p_emp={out['indeg_skew_p_emp']:.4g}"
          f"  (uniform null={sk_u.mean():.2f}, z={out['indeg_skew_z_uniform']:.1f})")
    print(f"  asymmetry:       obs={asym_o:.2f}  size-prop null={ay_p.mean():.2f}±{ay_p.std():.2f}  z={out['asymmetry_z']:.1f}  p_emp={out['asymmetry_p_emp']:.4g}")
    print(f"  modularity:      obs={mod_o:.2f}  null={mod_n:.2f}±{mod_s:.2f}  z={mod_z:.1f}  p_emp={out['modularity_p_emp']:.4g}")
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"output/analysis/confusion_null_{args.model}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

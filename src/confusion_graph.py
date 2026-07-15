"""Story E: the speaker-confusion graph. For each query (speaker A) the nearest
DIFFERENT-speaker match (speaker B) gives a directed edge A->B. We analyse:
  - in-degree (hub speakers everyone is confused WITH) and its skew
  - asymmetry (A->B but not B->A)
  - communities of mutually-confusable speakers (voice "types")
  - correlation of hub in-degree with gender / fame

Recomputes nearest-impostor speaker from the embeddings (agency-only, AS-norm-ish
centered cosine). Usage: python -m src.confusion_graph --model ecapa
"""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

EMB = Path("output/embeddings"); AN = Path("output/analysis")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ecapa")
    ap.add_argument("--agency-only", action="store_true", default=True)
    args = ap.parse_args()
    d = np.load(EMB / f"{args.model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32); spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    X = emb[keep]; spk = spk[keep]
    X = X - X.mean(0, keepdims=True); X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    Xt = torch.from_numpy(X).cuda(); n = len(spk)
    _, codes = np.unique(spk, return_inverse=True); ct = torch.from_numpy(codes).cuda()
    ids = list(dict.fromkeys(spk.tolist()))  # speaker order
    neg = torch.tensor(-9.0, device="cuda")
    edges = Counter()  # (A_code, B_code) -> count
    for st in range(0, n, 2048):
        e = min(st + 2048, n); s = Xt[st:e] @ Xt.T
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        s = torch.where(same, neg, s)
        nb = s.argmax(1).cpu().numpy()
        for q, b in zip(range(st, e), nb):
            edges[(int(codes[q]), int(codes[b]))] += 1

    # speaker-level directed weights A->B (A confused-as B)
    spk_edges = defaultdict(int)
    for (a, b), c in edges.items():
        spk_edges[(a, b)] += c
    nspk = len(set(codes.tolist()))
    indeg = np.zeros(nspk); outdeg = np.zeros(nspk)
    for (a, b), c in spk_edges.items():
        indeg[b] += c; outdeg[a] += c
    # asymmetry: of speaker-pairs with A->B, how many also have B->A?
    pairset = {(a, b) for (a, b), c in spk_edges.items() if a != b}
    recip = sum(1 for (a, b) in pairset if (b, a) in pairset)
    asym = 1 - recip / max(len(pairset), 1)

    # top hubs (highest in-degree = most speakers/queries land on them)
    code2name = {}
    for nm in json.load(open("data/registry/speakers.jsonl".replace("jsonl", "jsonl"))) if False else []:
        pass
    names = {}
    for l in open("data/registry/speakers.jsonl", encoding="utf-8"):
        s = json.loads(l); names[s["speaker_id"]] = s["name"]
    gender = json.load(open("data/registry/gender.json", encoding="utf-8"))
    code2spk = {int(c): ids[i] for i, c in enumerate([codes[np.where(spk == s)[0][0]] for s in ids])}
    top = np.argsort(-indeg)[:12]

    # communities via greedy modularity on the symmetrized confusion graph
    communities = None
    try:
        import networkx as nx
        G = nx.Graph()
        for (a, b), c in spk_edges.items():
            if a != b:
                G.add_edge(a, b, weight=G.get_edge_data(a, b, {}).get("weight", 0) + c)
        comm = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        communities = sorted([len(c) for c in comm], reverse=True)[:10]
        modularity = nx.community.modularity(G, comm, weight="weight")
    except Exception as exc:
        modularity = None
        print(f"  (networkx community skip: {exc})")

    # hub in-degree vs gender
    g_indeg = defaultdict(list)
    for c in range(nspk):
        sid = code2spk.get(c)
        g_indeg[gender.get(sid, "unknown")].append(indeg[c])

    res = {
        "model": args.model, "n_speakers": nspk, "n_edges_pairs": len(pairset),
        "indegree_skew": float(stats.skew(indeg)),
        "indegree_max_over_mean": float(indeg.max() / indeg.mean()),
        "asymmetry_fraction": float(asym),
        "n_communities": (len(communities) if communities else None),
        "community_sizes_top": communities,
        "modularity": (float(modularity) if modularity is not None else None),
        "top_hubs": [{"name": names.get(code2spk.get(int(c)), code2spk.get(int(c))),
                      "in_degree": int(indeg[c]), "gender": gender.get(code2spk.get(int(c)), "?")}
                     for c in top],
        "mean_indegree_by_gender": {g: float(np.mean(v)) for g, v in g_indeg.items()},
    }
    print(f"#### {args.model} confusion graph ({nspk} speakers) ####")
    print(f"  in-degree skew={res['indegree_skew']:.2f}  max/mean={res['indegree_max_over_mean']:.1f}x")
    print(f"  asymmetry: {asym*100:.0f}% of confusion pairs are one-directional")
    if communities:
        print(f"  communities: {len(communities)} (top sizes {communities[:6]}), modularity={modularity:.2f}")
    print("  mean in-degree by gender:", {g: round(v, 1) for g, v in res["mean_indegree_by_gender"].items()})
    print("  top confusion hubs (who others are mistaken FOR):")
    for h in res["top_hubs"][:8]:
        print(f"    {h['name']:14s} in-deg={h['in_degree']:4d}  ({h['gender']})")
    AN.mkdir(parents=True, exist_ok=True); json.dump(res, open(AN / f"confusion_graph_{args.model}.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

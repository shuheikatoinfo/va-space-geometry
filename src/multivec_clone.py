"""Multi-vector (per-style) enrollment for the clone probe, vs centroid enrollment.
Each actor is enrolled with K real segment embeddings (not a single centroid); a
clone's score to an actor = MAX cosine over that actor's K vectors (so a clone of
any of the actor's styles can match). Detection threshold from real held-out trials.

Usage: python -m src.multivec_clone --clone-dir .../irodori --label irodori --model animeva
"""
from __future__ import annotations

import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, soundfile as sf

from src.clone_probe import va_actor_segments, eer_threshold
from src.embeddings import build_extractor

K = 8


def emb_file(ext, p):
    w, _ = sf.read(p, dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
    return np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)


def run(clone_dir, label, model):
    ext = build_extractor(model, "cuda")
    actors = va_actor_segments(); ids = list(actors); idx = {a: i for i, a in enumerate(ids)}
    # enrollment: K vectors/actor (centered+LN); galleries as a stacked matrix with owner ids
    mean = None
    vecs, owner = [], []
    raw = {a: np.stack([emb_file(ext, p) for p in actors[a][:K]]) for a in ids}
    allv = np.concatenate(list(raw.values()), 0); mean = allv.mean(0, keepdims=True)
    def norm(x): return (x - mean) / np.clip(np.linalg.norm(x - mean, axis=1, keepdims=True), 1e-9, None)
    for a in ids:
        v = norm(raw[a])
        vecs.append(v); owner += [idx[a]] * len(v)
    G = np.concatenate(vecs, 0); owner = np.array(owner)        # (sum K_a, D)
    Gt = torch.from_numpy(G).cuda(); ot = torch.from_numpy(owner).cuda()
    nA = len(ids)

    def actor_max(qn):  # qn: (m,D) normed -> (m,nA) max-cos per actor
        s = torch.from_numpy(qn).cuda() @ Gt.T
        out = torch.full((s.shape[0], nA), -9.0, device="cuda")
        out = out.index_reduce(1, ot, s, "amax", include_self=False) if hasattr(out, "index_reduce") else None
        if out is None:
            out = torch.full((s.shape[0], nA), -9.0, device="cuda")
            for j in range(nA):
                out[:, j] = s[:, owner == j].max(1).values
        return out.cpu().numpy()

    # threshold from real held-out segs (8..12) vs gallery
    gen, imp = [], []
    for a in ids:
        if len(actors[a]) <= K: continue
        q = norm(np.stack([emb_file(ext, p) for p in actors[a][K:K + 4]]))
        sc = actor_max(q)
        gen += list(sc[:, idx[a]]); imp += list(np.delete(sc, idx[a], 1).max(1))
    thr = eer_threshold(np.array(gen), np.array(imp))

    rows = []
    for f in sorted(Path(clone_dir).glob("clone_*__src*.wav")):
        m = re.match(r"clone_(.+)__src\d+", f.stem)
        if not m or m.group(1) not in idx: continue
        T = m.group(1)
        sc = actor_max(norm(emb_file(ext, f)[None]))[0]
        simT = sc[idx[T]]; near = ids[int(np.argmax(sc))]
        rows.append({"target": T, "det": int(simT >= thr), "false": int(near != T)})
    by = defaultdict(list)
    for r in rows: by[r["target"]].append(r)
    uts = list(by); rng = np.random.default_rng(0)
    def rate(rs, k): return float(np.mean([r[k] for r in rs])) if rs else float("nan")
    boot = defaultdict(list)
    for _ in range(1000):
        samp = rng.choice(uts, len(uts), replace=True)
        rs = [r for t in samp for r in by[t]]
        for k in ["det", "false"]: boot[k].append(rate(rs, k))
    res = {"label": label, "model": model, "enroll": f"multivec K={K}", "n": len(rows),
           "detectable": rate(rows, "det"), "false_attr": rate(rows, "false"),
           "ci_det": [float(np.percentile(boot["det"], 2.5)), float(np.percentile(boot["det"], 97.5))],
           "ci_false": [float(np.percentile(boot["false"], 2.5)), float(np.percentile(boot["false"], 97.5))]}
    print(f"  {label}/{model} MULTIVEC(K={K}): detect={res['detectable']*100:.1f}% CI[{res['ci_det'][0]*100:.0f}-{res['ci_det'][1]*100:.0f}]  "
          f"false-attr={res['false_attr']*100:.1f}% CI[{res['ci_false'][0]*100:.0f}-{res['ci_false'][1]*100:.0f}]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["irodori", "knnvc", "gptsovits"])
    ap.add_argument("--model", default="animeva")
    args = ap.parse_args()
    base = "/path/to/va-data/clones"
    out = {}
    print(f"=== multi-vector (K={K}) enrollment vs clones ({args.model}) ===")
    for lab in args.pairs:
        out[lab] = run(f"{base}/{lab}", lab, args.model)
    json.dump(out, open("output/analysis/multivec_clone.json", "w"), indent=2)


if __name__ == "__main__":
    main()

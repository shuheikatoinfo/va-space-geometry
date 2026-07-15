"""Matched comparison: is the voice-actor (VA) embedding space denser / more
hub-prone than control populations (JVS professionals, Common Voice general public)?

For a fair comparison we match the number of speakers and segments-per-speaker
across populations (hubness depends on N), then compute AS-norm-calibrated
rank-1 / EER / critical-margin / hubness for each. Same model, same backend.

Usage:
    python -m src.population_compare --model jxvector
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

EMB_DIR = Path("output/embeddings")
CTRL_DIR = Path("output/embeddings_control")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pop(model: str, pop: str):
    if pop == "va":
        d = np.load(EMB_DIR / f"{model}.npz", allow_pickle=True)
        spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"])
        keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
        return d["emb"][keep].astype(np.float32), spk[keep]
    d = np.load(CTRL_DIR / f"{pop}_{model}.npz", allow_pickle=True)
    return d["emb"].astype(np.float32), np.asarray(d["speaker_id"])


def match_subset(emb, spk, n_spk, m_seg, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, s in enumerate(spk):
        by[s].append(i)
    spks = [s for s in by if len(by[s]) >= m_seg]
    spks = list(rng.permutation(spks))[:n_spk]
    idx = []
    for s in spks:
        idx += list(rng.choice(by[s], m_seg, replace=False))
    idx = np.array(idx)
    return emb[idx], spk[idx]


def asnorm_eval(emb, spk, k=10, cohort_n=200):
    X = emb - emb.mean(0, keepdims=True)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    Xt = torch.from_numpy(X).to(DEVICE); n = len(spk)
    _, codes = np.unique(spk, return_inverse=True); ct = torch.from_numpy(codes).to(DEVICE)
    neg = torch.tensor(-9.0, device=DEVICE)
    cn = min(cohort_n, n - 1)
    mu = torch.empty(n, device=DEVICE); sd = torch.empty(n, device=DEVICE)
    for st in range(0, n, 2048):
        e = min(st + 2048, n); s = Xt[st:e] @ Xt.T
        sm = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        s = torch.where(sm, neg, s); top = s.topk(cn, 1).values
        mu[st:e] = top.mean(1); sd[st:e] = top.std(1).clamp_min(1e-6)
    sb = np.full(n, -9.0, np.float32); db = np.full(n, -9.0, np.float32); occ = np.zeros(n, np.int64)
    for st in range(0, n, 2048):
        e = min(st + 2048, n); s = Xt[st:e] @ Xt.T
        s = 0.5 * ((s - mu[st:e].unsqueeze(1)) / sd[st:e].unsqueeze(1) + (s - mu.unsqueeze(0)) / sd.unsqueeze(0))
        rows = torch.arange(st, e, device=DEVICE); s[torch.arange(e - st, device=DEVICE), rows] = neg
        sm = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        sb[st:e] = torch.where(sm, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(sm, neg, s).max(1).values.cpu().numpy()
        np.add.at(occ, s.topk(k, 1).indices.reshape(-1).cpu().numpy(), 1)
    has = sb > -8; m = (sb - db)[has]
    spread = m.std()
    # EER from genuine(sb) vs impostor(db)
    tgt, non = sb[has], db[has]
    sc = np.concatenate([tgt, non]); lb = np.concatenate([np.ones_like(tgt), np.zeros_like(non)])
    o = np.argsort(-sc); lb = lb[o]; P = lb.sum(); N = len(lb) - P
    tp = np.cumsum(lb); fp = np.cumsum(1 - lb); fnr = 1 - tp / P; fpr = fp / N
    eer = float((fnr + fpr)[np.argmin(np.abs(fnr - fpr))] / 2)
    return dict(n=n, rank1=float(np.mean(sb[has] > db[has])), eer=eer,
                misid=float(np.mean(m < 0)), critical=float(np.mean(np.abs(m) < 0.1 * spread)),
                hub=float(stats.skew(occ.astype(float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jxvector")
    ap.add_argument("--m-seg", type=int, default=8)
    args = ap.parse_args()
    pops = {}
    for p in ["va", "jvs", "cv"]:
        try:
            pops[p] = load_pop(args.model, p)
        except FileNotFoundError:
            print(f"(skip {p}: embeddings not found)")
    # match sizes pairwise to the smaller speaker count
    results = {}
    for label, (n_spk, seeds) in {"vs_JVS(100spk)": (100, range(3)), "vs_CV(matched)": (700, range(3))}.items():
        present = [p for p in ["va", "jvs", "cv"] if p in pops]
        target = "jvs" if "JVS" in label else "cv"
        if target not in pops:
            continue
        cap = min(len(set(pops[target][1])), n_spk)
        print(f"\n### {label}: matched to {cap} speakers x {args.m_seg} segs ({args.model}) ###")
        for p in ["va", target]:
            accs = defaultdict(list)
            for sd in seeds:
                emb, spk = match_subset(*pops[p], cap, args.m_seg, seed=sd)
                r = asnorm_eval(emb, spk)
                for kk in ["rank1", "eer", "misid", "critical", "hub"]:
                    accs[kk].append(r[kk])
            mean = {kk: float(np.mean(v)) for kk, v in accs.items()}
            print(f"  {p.upper():4s} N={r['n']:5d}  rank1={mean['rank1']*100:5.1f}%  EER={mean['eer']*100:4.1f}%  "
                  f"misid={mean['misid']*100:4.1f}%  crit={mean['critical']*100:4.1f}%  hub={mean['hub']:.2f}")
            results[f"{label}:{p}"] = mean
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(results, open(f"output/analysis/popcompare_{args.model}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

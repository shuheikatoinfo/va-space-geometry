"""Standard trial-based verification metrics (EER, minDCF, minCllr) and a test of
whether the open-set misidentification floor resists discriminative RE-RANKING
back-ends (LDA, WCCN) in addition to AS-norm.

Speaker-disjoint split: LDA/WCCN are trained on TRAIN speakers, all metrics on
EVAL speakers (agency-only). Backends: cosine (center+LN), AS-norm, LDA+cos,
WCCN+cos. We report pairwise EER/minDCF/minCllr (trial-based) AND the open-set 1:N
misID (best-same vs best-diff) under each backend.

Usage: python -m src.verification_metrics --models ecapa ens_sv4 animeva
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.isotonic import IsotonicRegression

EMB = Path("output/embeddings"); AN = Path("output/analysis")
RNG = np.random.default_rng(0)


def ln(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def eer(tar, non):
    s = np.concatenate([tar, non]); y = np.concatenate([np.ones_like(tar), np.zeros_like(non)])
    o = np.argsort(-s); y = y[o]
    P, N = y.sum(), len(y) - y.sum()
    fnr = 1 - np.cumsum(y) / P; fpr = np.cumsum(1 - y) / N
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def mindcf(tar, non, p, cmiss=1, cfa=1):
    thr = np.sort(np.concatenate([tar, non]))
    pmiss = np.searchsorted(np.sort(tar), thr) / len(tar)
    pfa = 1 - np.searchsorted(np.sort(non), thr) / len(non)
    dcf = cmiss * p * pmiss + cfa * (1 - p) * pfa
    return float(dcf.min() / min(cmiss * p, cfa * (1 - p)))


def mincllr(tar, non):
    # optimal (PAV) calibration -> minCllr (BOSARIS); discrimination floor
    s = np.concatenate([tar, non]); y = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])
    p = IsotonicRegression(out_of_bounds="clip").fit_transform(s, y)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lr = (p / (1 - p)) * (len(non) / len(tar))  # posterior -> LR (remove empirical prior)
    lr_t, lr_n = lr[: len(tar)], lr[len(tar):]
    return float(0.5 * (np.mean(np.log2(1 + 1 / lr_t)) + np.mean(np.log2(1 + lr_n))))


def trials(emb, spk, n=40000):
    by = defaultdict(list)
    for i, s in enumerate(spk):
        by[s].append(i)
    multi = [s for s in by if len(by[s]) >= 2]
    tar = []
    for _ in range(n):
        s = multi[RNG.integers(len(multi))]; a, b = RNG.choice(by[s], 2, replace=False); tar.append((a, b))
    non = []
    spks = list(by)
    for _ in range(n):
        s1, s2 = RNG.choice(len(spks), 2, replace=False)
        non.append((RNG.choice(by[spks[s1]]), RNG.choice(by[spks[s2]])))
    return np.array(tar), np.array(non)


def openset_misid(Xn, spk):
    # NB: despite the historical name, this is the CLOSED-SET rank-1 misID (= 1 - rank-1):
    # every probe's speaker is enrolled; there is no reject class. The paper labels it
    # "closed-set rank-1 identification error" (Appendix A). Kept as `openset_misid` only
    # for artifact/JSON-key stability across the released outputs.
    import torch
    Xt = torch.from_numpy(Xn.astype(np.float32)).cuda(); n = len(spk)
    _, c = np.unique(spk, return_inverse=True); ct = torch.from_numpy(c).cuda()
    neg = torch.tensor(-9.0, device="cuda"); sb = np.full(n, -9.0); db = np.full(n, -9.0)
    for st in range(0, n, 2048):
        e = min(st + 2048, n); s = Xt[st:e] @ Xt.T
        rows = torch.arange(st, e, device="cuda"); s[torch.arange(e - st, device="cuda"), rows] = neg
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        sb[st:e] = torch.where(same, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same, neg, s).max(1).values.cpu().numpy()
    has = sb > -8
    return float(np.mean(sb[has] < db[has]))


def backend_transform(Xtr, ytr, Xev, kind):
    if kind == "cosine":
        return ln(Xev - Xtr.mean(0, keepdims=True))
    if kind == "lda":
        # Genuine dimensionality-reducing re-ranker. We cap n_components well below
        # the embedding dim so LDA is a DISTINCT operation from WCCN: under cosine a
        # full-rank LDA (whiten+rotate) equals full-rank WCCN, so keeping LDA full
        # rank would double-count one back-end (reviewer note). 150 < 192 (smallest
        # single-encoder dim), so LDA always reduces.
        nc = min(150, len(set(ytr)) - 1)
        lda = LinearDiscriminantAnalysis(n_components=nc).fit(Xtr, ytr)
        return ln(lda.transform(Xev))
    if kind == "wccn":
        m = Xtr.mean(0, keepdims=True); Xc = Xtr - m
        D = Xtr.shape[1]
        W = np.zeros((D, D))
        for s in set(ytr):
            xs = Xc[ytr == s]
            if len(xs) > 1:
                W += (xs - xs.mean(0)).T @ (xs - xs.mean(0))
        W = W / len(Xtr)
        # RELATIVE ridge (scaled to the trace), so it is a small regularizer rather
        # than a constant that dominates low-variance dims at high D (reviewer note:
        # a fixed 1e-4 exceeded within-var in 45-62% of ensemble dims).
        W = W + 1e-2 * (np.trace(W) / D) * np.eye(D)
        # eigh-based inverse sqrt (exact for symmetric PSD; exposes small eigenvalues
        # instead of hiding them behind sqrtm(...).real).
        wval, wvec = np.linalg.eigh((W + W.T) / 2)
        wval = np.clip(wval, 1e-12, None)
        Wm = (wvec / np.sqrt(wval)) @ wvec.T
        return ln((Xev - m) @ Wm)
    raise ValueError(kind)


def run(model, seed=0):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32); spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    emb, spk = emb[keep], spk[keep]
    sp_uniq = np.array(sorted(set(spk.tolist())))
    np.random.default_rng(seed).shuffle(sp_uniq)
    tr_sp = set(sp_uniq[: int(0.55 * len(sp_uniq))]);
    trm = np.array([s in tr_sp for s in spk]); evm = ~trm
    Xtr, ytr = emb[trm], spk[trm]; Xev, yev = emb[evm], spk[evm]
    ta, no = trials(Xev, yev)
    out = {"model": model, "n_eval_spk": int(len(set(yev.tolist())))}
    for kind in ["cosine", "lda", "wccn"]:
        Z = backend_transform(Xtr, ytr, Xev, kind)
        tar = np.sum(Z[ta[:, 0]] * Z[ta[:, 1]], 1); non = np.sum(Z[no[:, 0]] * Z[no[:, 1]], 1)
        out[kind] = {"EER": eer(tar, non), "minDCF_0.01": mindcf(tar, non, 0.01),
                     "minDCF_0.05": mindcf(tar, non, 0.05), "minCllr": mincllr(tar, non),
                     "openset_misID": openset_misid(Z, yev)}
    print(f"\n#### {model} (eval speakers={out['n_eval_spk']}) ####")
    print(f"  {'backend':8s} {'EER%':>6s} {'minDCF.01':>9s} {'minDCF.05':>9s} {'minCllr':>8s} {'1:N misID%':>10s}")
    for k in ["cosine", "lda", "wccn"]:
        r = out[k]
        print(f"  {k:8s} {r['EER']*100:6.1f} {r['minDCF_0.01']:9.3f} {r['minDCF_0.05']:9.3f} {r['minCllr']:8.3f} {r['openset_misID']*100:10.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["ecapa", "ens_sv4", "animeva"])
    ap.add_argument("--n-splits", type=int, default=5, help="average over N speaker-disjoint splits")
    args = ap.parse_args()
    res = {}
    for m in args.models:
        runs = [run(m, seed=s) for s in range(args.n_splits)]
        agg = {"model": m, "n_eval_spk": runs[0]["n_eval_spk"], "n_splits": args.n_splits}
        for kind in ["cosine", "lda", "wccn"]:
            agg[kind] = {}
            for metric in runs[0][kind]:
                vals = [r[kind][metric] for r in runs]
                agg[kind][metric] = float(np.mean(vals))
                agg[kind][metric + "_std"] = float(np.std(vals))
        res[m] = agg
        c = agg["cosine"]; print(f"{m:9s} cosine misID {c['openset_misID']*100:.1f}±{c['openset_misID_std']*100:.1f}% "
                                  f"EER {c['EER']*100:.1f}% (avg of {args.n_splits} splits)")
    AN.mkdir(parents=True, exist_ok=True); json.dump(res, open(AN / "verification_metrics.json", "w"), indent=2)
    print(f"\n-> {AN/'verification_metrics.json'}")


if __name__ == "__main__":
    main()

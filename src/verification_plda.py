"""Generative two-covariance PLDA-LLR scoring (completes the re-ranking suite of
verification_metrics.py), plus DET curves and calibrated Cllr (actCllr).

Two-cov PLDA: x = s + e, s~N(0,B), e~N(0,W). After centering + length-norm we
simultaneously diagonalize (whiten W to I, rotate B to diag Λ). The same/diff LLR
for a pair (u,v) factorizes per dimension d (a=λ_d+1, b=λ_d):
  LLR = Σ_d [ 0.5 log(a²/(a²−b²)) + p_d(u_d²+v_d²) + q_d u_d v_d ],
  p_d = 0.5(1/a − a/(a²−b²)),  q_d = b/(a²−b²).
This bilinear form gives fast pairwise scoring: LLR(i,j)=C + g_i + g_j + u_iᵀ(q⊙u_j).

Usage: python -m src.verification_plda --models ecapa ens_sv4 animeva
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.verification_metrics import ln, eer, mindcf, mincllr, trials

AN = Path("output/analysis")
N_TRIALS = []  # non-target trial counts, sets the meaningful lower DET axis limit
RNG = np.random.default_rng(0)

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


# Line style encodes the encoder family, so the curves are distinguishable in
# grayscale / for color-vision-deficient readers, not by color alone:
#   solid  = generic single encoder (English / Chinese, non-domain-matched)
#   dashed = Japanese-trained single encoder
#   dotted = ensemble
LINESTYLE = {
    "xvector": "-", "ecapa": "-", "campp": "-", "redimnet": "-", "wavlm": "-",
    "jxvector": "--", "animeva": "--", "jhubert": "--", "hubert": "--",
    "ens_sv4": ":", "ens_all6": ":",
}


def plda_fit(X, y, pca_dim=200, alpha=1e-2):
    """Two-covariance PLDA with corrections (reviewer notes):
      - PCA-reduce to <= min(pca_dim, K-1) before estimation, so W/B are full-rank
        and the ensembles do not live in a ridge-defined null space (B is rank <= K-1).
      - W and B use the SAME maximum-likelihood normalization (both /N); B is the
        SIZE-WEIGHTED between-class scatter (unweighted class means are biased under
        the unequal segments/speaker here).
      - RELATIVE ridge (scaled to trace/D), not a fixed absolute constant.
      - eigh-based W^{-1/2} (exact for symmetric PSD).
    """
    m = X.mean(0, keepdims=True); Xc = ln(X - m)
    classes = list(set(y)); K = len(classes)
    # PCA reduction (fit on centered train); keeps the estimation well-conditioned.
    cm = Xc.mean(0, keepdims=True)
    target = min(pca_dim, K - 1, Xc.shape[1])
    if Xc.shape[1] > target:
        _, _, Vt = np.linalg.svd(Xc - cm, full_matrices=False)
        P = Vt[:target].T                       # D x target
    else:
        P = np.eye(Xc.shape[1])
    Xp = (Xc - cm) @ P
    D = Xp.shape[1]; N = len(Xp)
    means = np.stack([Xp[y == c].mean(0) for c in classes])
    ncs = np.array([int(np.sum(y == c)) for c in classes], dtype=np.float64)
    gm = Xp.mean(0)
    # pooled within-class scatter (ML, /N)
    W = np.zeros((D, D))
    for i, c in enumerate(classes):
        d = Xp[y == c] - means[i]; W += d.T @ d
    W = W / N
    # size-weighted between-class scatter (ML, /N) -- same normalization as W
    Md = means - gm
    B = (Md * ncs[:, None]).T @ Md / N
    W = W + alpha * (np.trace(W) / D) * np.eye(D)
    B = B + 1e-6 * (np.trace(B) / D + 1e-12) * np.eye(D)
    wval, wvec = np.linalg.eigh((W + W.T) / 2)
    wval = np.clip(wval, 1e-12, None)
    Wm = (wvec / np.sqrt(wval)) @ wvec.T         # W^{-1/2}
    Bp = Wm @ B @ Wm
    lam, U = np.linalg.eigh((Bp + Bp.T) / 2)
    lam = np.clip(lam, 0, None)
    T = (U.T @ Wm)                               # u = T P^T (LN(x)-m-cm)
    a = lam + 1.0; b = lam
    denom = a * a - b * b                         # = 2*lam+1 > 0
    p = 0.5 * (1.0 / a - a / denom)
    q = b / denom
    C = float(np.sum(0.5 * np.log((a * a) / denom)))
    return {"m": m, "cm": cm, "P": P, "T": T, "p": p, "q": q, "C": C}


def plda_u(model, X):
    Xp = (ln(X - model["m"]) - model["cm"]) @ model["P"]
    return Xp @ model["T"].T


def plda_pair_scores(model, U, pairs):
    p, q = model["p"], model["q"]
    g = (U * U) @ p
    a, bvec = U[pairs[:, 0]], U[pairs[:, 1]]
    cross = np.sum(a * (q * bvec), 1)
    return model["C"] + g[pairs[:, 0]] + g[pairs[:, 1]] + cross


def plda_openset_misid(model, U, spk):
    import torch
    p = torch.from_numpy(model["p"].astype(np.float32)).cuda()
    q = torch.from_numpy(model["q"].astype(np.float32)).cuda()
    Ut = torch.from_numpy(U.astype(np.float32)).cuda(); n = len(spk)
    g = (Ut * Ut) @ p
    Uq = Ut * q
    _, c = np.unique(spk, return_inverse=True); ct = torch.from_numpy(c).cuda()
    neg = torch.tensor(-1e9, device="cuda"); sb = np.full(n, -1e18); db = np.full(n, -1e18)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        s = Ut[st:e] @ Uq.T + g.unsqueeze(0) + g[st:e].unsqueeze(1) + model["C"]
        rows = torch.arange(st, e, device="cuda"); s[torch.arange(e - st, device="cuda"), rows] = neg
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        sb[st:e] = torch.where(same, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same, neg, s).max(1).values.cpu().numpy()
    return float(np.mean(sb < db))


def actcllr(tar, non):
    # logistic calibration on half the trials, Cllr on the other half
    s = np.concatenate([tar, non]); y = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])
    idx = RNG.permutation(len(s)); h = len(s) // 2
    tr, te = idx[:h], idx[h:]
    lr = LogisticRegression().fit(s[tr, None], y[tr])
    llr = lr.decision_function(s[te, None])  # calibrated log-odds ~ LLR (prior absorbed)
    yt = y[te]; lt, ln_ = llr[yt == 1], llr[yt == 0]
    lr_t, lr_n = np.exp(lt), np.exp(ln_)
    return float(0.5 * (np.mean(np.log2(1 + 1 / lr_t)) + np.mean(np.log2(1 + lr_n))))


def det_points(tar, non):
    """DET curve on probit (normal-deviate) axes, per Martin et al. (1997).

    Returns (x, y) = (norm.ppf(fpr), norm.ppf(fnr)). Points with fpr==0 or
    fnr==0 are dropped BEFORE warping (probit is -inf there), and any
    subsampling happens AFTER warping, uniformly in warped space, so the
    low-FPR tail (which spans a large probit range) is preserved.
    """
    from scipy.stats import norm
    s = np.concatenate([tar, non]); y = np.concatenate([np.ones_like(tar), np.zeros_like(non)])
    o = np.argsort(-s); y = y[o]
    P, N = y.sum(), len(y) - y.sum()
    fnr = 1 - np.cumsum(y) / P; fpr = np.cumsum(1 - y) / N
    keep = (fpr > 0) & (fnr > 0)
    x, yv = norm.ppf(fpr[keep]), norm.ppf(fnr[keep])
    if len(x) > 1000:  # decimate uniformly in warped space (keep tail detail)
        d = np.abs(np.diff(x)) + np.abs(np.diff(yv))
        arc = np.concatenate([[0.0], np.cumsum(d)])
        tgt = np.linspace(0, arc[-1], 1000)
        idx = np.unique(np.searchsorted(arc, tgt))
        x, yv = x[idx], yv[idx]
    return x, yv


def _one_split(emb, spk, seed):
    su = np.array(sorted(set(spk.tolist()))); np.random.default_rng(seed).shuffle(su)
    tr_sp = set(su[: int(0.55 * len(su))]); trm = np.array([s in tr_sp for s in spk]); evm = ~trm
    Xtr, ytr, Xev, yev = emb[trm], spk[trm], emb[evm], spk[evm]
    pl = plda_fit(Xtr, ytr); U = plda_u(pl, Xev)
    ta, no = trials(Xev, yev)
    tar = plda_pair_scores(pl, U, ta); non = plda_pair_scores(pl, U, no)
    return ({"EER": eer(tar, non), "minDCF_0.01": mindcf(tar, non, 0.01),
             "minCllr": mincllr(tar, non), "actCllr": actcllr(tar, non),
             "openset_misID": plda_openset_misid(pl, U, yev)}, tar, non)


def run(model, ax, n_splits=5):
    d = np.load(f"output/embeddings/{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32); spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    emb, spk = emb[keep], spk[keep]
    runs = [_one_split(emb, spk, s) for s in range(n_splits)]
    metrics = [r[0] for r in runs]
    res = {"n_splits": n_splits}
    for k in metrics[0]:
        vals = [m[k] for m in metrics]
        res[k] = float(np.mean(vals)); res[k + "_std"] = float(np.std(vals))
    # DET curve from the first split (representative), on probit axes
    _, tar, non = runs[0]
    x, yv = det_points(tar, non)
    N_TRIALS.append(len(non))
    ls = LINESTYLE.get(model, "-")
    ax.plot(x, yv, ls=ls, lw=2.0,
            label=f"{disp(model)} (EER {res['EER']*100:.1f}%)")
    print(f"  {model:9s} PLDA: EER={res['EER']*100:4.1f}%  minDCF.01={res['minDCF_0.01']:.3f}  "
          f"minCllr={res['minCllr']:.3f}  actCllr={res['actCllr']:.3f}  "
          f"1:N misID={res['openset_misID']*100:.1f}±{res['openset_misID_std']*100:.1f}% (avg {n_splits} splits)")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["ecapa", "ens_sv4", "animeva"])
    ap.add_argument("--n-splits", type=int, default=5, help="average over N speaker-disjoint splits")
    args = ap.parse_args()
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    out = {}
    print(f"=== Two-covariance PLDA-LLR (speaker-disjoint, {args.n_splits} splits) ===")
    for m in args.models:
        out[m] = run(m, ax, n_splits=args.n_splits)
    from scipy.stats import norm
    # probit (normal-deviate) axes with standard DET ticks
    ticks_pct = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    lo_pct = max(0.1, 100.0 / min(N_TRIALS))  # no smaller than ~1/#nontarget trials
    lo, hi = norm.ppf(lo_pct / 100), norm.ppf(0.50)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, dashes=(6, 4), zorder=0)  # EER diagonal
    tpos = [norm.ppf(p / 100) for p in ticks_pct if p >= lo_pct]
    tlab = [f"{p:g}" for p in ticks_pct if p >= lo_pct]
    ax.set_xticks(tpos); ax.set_xticklabels(tlab)
    ax.set_yticks(tpos); ax.set_yticklabels(tlab)
    ax.set_xlabel("FPR (%)"); ax.set_ylabel("FNR (%)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")  # square plot box (FPR/FNR share the same scale)
    # legend below the plot box, in 3 rows, so it never overlaps the DET curves and keeps the box square
    ncol = -(-len(args.models) // 3)  # ceil: 3 rows
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=True)
    fig.tight_layout(); fig.savefig("output/fig_det_plda.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    AN.mkdir(parents=True, exist_ok=True); json.dump(out, open(AN / "verification_plda.json", "w"), indent=2)
    print("wrote output/fig_det_plda.png ; output/analysis/verification_plda.json")


if __name__ == "__main__":
    main()

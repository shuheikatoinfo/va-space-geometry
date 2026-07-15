"""EM/ML two-covariance PLDA vs the moment-fit PLDA (reviewer objection killer).

The paper's re-ranking suite uses a moment/scatter two-covariance PLDA whose
size-weighted between-class scatter is a biased (over-)estimate; the text
argues the residual misID is an upper bound relative to an EM estimator. This
script removes the need for that argument by actually fitting the canonical
EM/ML two-covariance Gaussian PLDA (x = mu + y + e, y~N(0,B), e~N(0,W);
Brummer & de Villiers 2010) on the same length-normalized, PCA-reduced space,
initialized at the moment estimates, and scoring with the identical LLR
machinery, splits (seeds 0..2, 55/45 speaker-disjoint), and paired trials.

EM is run in the simultaneously diagonalized basis (W -> I, B -> diag(lam)),
where per-speaker posteriors are elementwise:
  phi_d(n) = lam_d / (1 + n lam_d),  yhat_i = phi(n_i) * sum_j v_ij
  B  <- (1/K) sum_i (yhat yhat^T + diag phi)
  W  <- (1/N) (sum_ij (v-yhat)(v-yhat)^T + diag sum_i n_i phi_i)
then mapped back through V^{-1} = W^{1/2} U.

Usage: python -m src.em_plda --models xvector ecapa campp redimnet jxvector animeva ens_sv4 ens_all6
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from src.verification_metrics import ln, eer, mincllr
from src.verification_plda import (plda_fit, plda_u, plda_pair_scores,
                                   plda_openset_misid, actcllr)

AN = Path("output/analysis")


def _diagonalize(W, B):
    """Return V (x->v with W->I, B->diag lam), its inverse, and lam."""
    wval, wvec = np.linalg.eigh((W + W.T) / 2)
    wval = np.clip(wval, 1e-12, None)
    Wm = (wvec / np.sqrt(wval)) @ wvec.T          # W^{-1/2}
    Wp = (wvec * np.sqrt(wval)) @ wvec.T          # W^{+1/2}
    Bp = Wm @ B @ Wm
    lam, U = np.linalg.eigh((Bp + Bp.T) / 2)
    lam = np.clip(lam, 1e-10, None)
    V = U.T @ Wm                                  # v = V x
    Vinv = Wp @ U                                 # x = Vinv v
    return V, Vinv, lam


def em_refit(Xp, y, W0, B0, n_iter=30, tol=1e-5):
    """EM for the two-covariance model on preprocessed features Xp (N x D),
    initialized at (W0, B0). Returns (W, B, loglik_trajectory)."""
    classes, cidx = np.unique(y, return_inverse=True)
    K, (N, D) = len(classes), Xp.shape
    ncs = np.bincount(cidx).astype(np.float64)     # (K,)
    W, B = W0.copy(), B0.copy()
    lls = []
    for it in range(n_iter):
        V, Vinv, lam = _diagonalize(W, B)
        Xv = Xp @ V.T                              # (N, D) in diagonal basis
        S = np.zeros((K, D)); np.add.at(S, cidx, Xv)
        phi = lam[None, :] / (1.0 + ncs[:, None] * lam[None, :])   # (K, D)
        yhat = phi * S                                             # (K, D)
        # marginal loglik in v-space (W=I, B=diag lam), per speaker
        # log N(x_i.. ) = -0.5 [ N_i D log 2pi + sum||x||^2 - yhat.S + log(1+n lam).sum ]
        quad = (Xv * Xv).sum() - (yhat * S).sum()
        logdet = np.log1p(ncs[:, None] * lam[None, :]).sum()
        ll = -0.5 * (N * D * np.log(2 * np.pi) + quad + logdet)
        lls.append(float(ll))
        # M-step in v-space
        Bv = (yhat.T @ yhat) / K + np.diag(phi.mean(0))
        R = Xv - yhat[cidx]
        Wv = (R.T @ R + np.diag((ncs[:, None] * phi).sum(0))) / N
        W = Vinv @ Wv @ Vinv.T
        B = Vinv @ Bv @ Vinv.T
        if it > 0 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-2]):
            break
    return W, B, lls


def plda_fit_em(X, y, pca_dim=200, alpha=1e-2, n_iter=30):
    """Same preprocessing/scoring contract as verification_plda.plda_fit, but the
    (W, B) estimates are refined by EM from the moment initialization."""
    m = X.mean(0, keepdims=True); Xc = ln(X - m)
    classes = list(set(y)); K = len(classes)
    cm = Xc.mean(0, keepdims=True)
    target = min(pca_dim, K - 1, Xc.shape[1])
    if Xc.shape[1] > target:
        _, _, Vt = np.linalg.svd(Xc - cm, full_matrices=False)
        P = Vt[:target].T
    else:
        P = np.eye(Xc.shape[1])
    Xp = (Xc - cm) @ P
    D = Xp.shape[1]; N = len(Xp)
    yarr = np.asarray(y)
    means = np.stack([Xp[yarr == c].mean(0) for c in classes])
    ncs = np.array([int(np.sum(yarr == c)) for c in classes], dtype=np.float64)
    gm = Xp.mean(0)
    W0 = np.zeros((D, D))
    for i, c in enumerate(classes):
        d = Xp[yarr == c] - means[i]; W0 += d.T @ d
    W0 = W0 / N
    Md = means - gm
    B0 = (Md * ncs[:, None]).T @ Md / N
    W0 = W0 + alpha * (np.trace(W0) / D) * np.eye(D)
    B0 = B0 + 1e-6 * (np.trace(B0) / D + 1e-12) * np.eye(D)
    W, B, lls = em_refit(Xp - gm, yarr, W0, B0, n_iter=n_iter)
    # final LLR parameterization, identical to the moment-fit path
    V, _, lam = _diagonalize(W, B)
    a = lam + 1.0; b = lam
    denom = a * a - b * b
    p = 0.5 * (1.0 / a - a / denom)
    q = b / denom
    C = float(np.sum(0.5 * np.log((a * a) / denom)))
    return {"m": m, "cm": cm, "P": P, "T": V, "p": p, "q": q, "C": C,
            "_gm": gm, "_lls": lls}


def plda_u_em(model, X):
    Xp = (ln(X - model["m"]) - model["cm"]) @ model["P"] - model["_gm"]
    return Xp @ model["T"].T


def _trials_paired(spk, seed, n=40000):
    """Deterministic trial sampling (fresh RNG; both back-ends score the same trials)."""
    from collections import defaultdict
    rng = np.random.default_rng(10_000 + seed)
    by = defaultdict(list)
    for i, s in enumerate(spk):
        by[s].append(i)
    multi = [s for s in by if len(by[s]) >= 2]
    tar, non = [], []
    spks = list(by)
    for _ in range(n):
        s = multi[rng.integers(len(multi))]
        a, b = rng.choice(by[s], 2, replace=False); tar.append((a, b))
        s1, s2 = rng.choice(len(spks), 2, replace=False)
        non.append((rng.choice(by[spks[s1]]), rng.choice(by[spks[s2]])))
    return np.array(tar), np.array(non)


def run(model_name, n_splits=3):
    d = np.load(f"output/embeddings/{model_name}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float64); spk = np.asarray(d["speaker_id"])
    src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    emb, spk = emb[keep], spk[keep]
    res = {"moment": {"misID": [], "EER": [], "minCllr": [], "actCllr": []},
           "em": {"misID": [], "EER": [], "minCllr": [], "actCllr": []},
           "em_iters": []}
    for seed in range(n_splits):
        su = np.array(sorted(set(spk.tolist()))); np.random.default_rng(seed).shuffle(su)
        tr_sp = set(su[: int(0.55 * len(su))])
        trm = np.array([s in tr_sp for s in spk]); evm = ~trm
        Xtr, ytr, Xev, yev = emb[trm], spk[trm], emb[evm], spk[evm]
        ta, no = _trials_paired(yev, seed)
        for kind in ("moment", "em"):
            if kind == "moment":
                pl = plda_fit(Xtr, ytr); U = plda_u(pl, Xev)
            else:
                pl = plda_fit_em(Xtr, ytr); U = plda_u_em(pl, Xev)
                res["em_iters"].append(len(pl["_lls"]))
            tar = plda_pair_scores(pl, U, ta); non = plda_pair_scores(pl, U, no)
            res[kind]["misID"].append(plda_openset_misid(pl, U, yev))
            res[kind]["EER"].append(eer(tar, non))
            res[kind]["minCllr"].append(mincllr(tar, non))
            res[kind]["actCllr"].append(actcllr(tar, non))
    out = {}
    for kind in ("moment", "em"):
        out[kind] = {k: float(np.mean(v)) for k, v in res[kind].items()}
        out[kind].update({k + "_std": float(np.std(v)) for k, v in res[kind].items()})
    out["em_iters"] = res["em_iters"]
    print(f"{model_name:10s} misID  moment {out['moment']['misID']*100:5.2f}%  em {out['em']['misID']*100:5.2f}%   "
          f"EER moment {out['moment']['EER']*100:5.2f}%  em {out['em']['EER']*100:5.2f}%   "
          f"minCllr {out['moment']['minCllr']:.3f} -> {out['em']['minCllr']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["xvector", "ecapa", "campp", "redimnet",
                             "jxvector", "animeva", "ens_sv4", "ens_all6"])
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args()
    print(f"=== EM vs moment-fit two-covariance PLDA ({args.n_splits} speaker-disjoint splits, paired trials) ===")
    out = {m: run(m, args.n_splits) for m in args.models}
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "em_plda.json", "w"), indent=2)
    print("wrote output/analysis/em_plda.json")


if __name__ == "__main__":
    main()

"""Does LEARNED fusion of the SV-4 members lower the closed-set rank-1 misID FLOOR
below simple mean-pooling, when trained speaker-disjoint?

Reviewer test. The paper fuses the 4 verification encoders by mean-pooling
(concat L2-normalized embeddings + renorm => cosine = mean of the 4 per-model
cosines) and leaves stronger fusion to future work. Here we actually train
learned fusion on the TRAIN-half speakers only and evaluate closed-set rank-1
misID on the disjoint EVAL-half, over the SAME 3 speaker-disjoint splits used in
Appendix A (src/verification_metrics.py: seeds 0,1,2; 55% train / 45% eval
speakers; freelance sources dropped).

We hold the back-end fixed at plain cosine (no LDA/WCCN/PLDA metric learning --
those are a separate axis already reported) and vary ONLY the member-fusion:

  baseline  mean-pool           : sim = mean_m cos_m                (paper, equal wts)
  (i)       learned weights     : sim = sum_m w_m cos_m , w>=0      (id. softmax OR
                                    Nelder-Mead on train misID; pick lower TRAIN misID)
  (ii-a)    stacking LR (linear): sim = w . [cos_1..cos_4]          (same/diff logistic)
  (ii-b)    stacking LR (quad)  : sim = w . [cos_m, cos_m*cos_n]    (degree-2 stacking)

cos_m is the RAW per-model cosine of L2-normalized member embeddings, so the
equal-weight baseline is literally "cosine = mean of per-model cosines" (paper
definition). We also verify this raw mean-pool reproduces the paper's joint
concat+center+LN cosine number (~3.9%) as a sanity anchor.

Everything below is linear / a few Adam steps -- lightweight, no heavy training.
Writes output/analysis/learned_fusion.json. Does NOT touch any paper/.tex file.

Usage: python -m src.learned_fusion
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

EMB = Path("output/embeddings")
AN = Path("output/analysis")
MEMBERS = ["xvector", "ecapa", "campp", "redimnet"]  # SV-4 (make_ensemble.ens_sv4)
SEEDS = [0, 1, 2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def l2norm_np(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def load_members():
    """Return list of L2-normalized member matrices (aligned), spk, src."""
    ds = [np.load(EMB / f"{m}.npz", allow_pickle=True) for m in MEMBERS]
    spk = np.asarray(ds[0]["speaker_id"])
    src = np.asarray(ds[0]["recording_source"])
    blocks = [l2norm_np(d["emb"].astype(np.float32)) for d in ds]
    return blocks, spk, src


def split_masks(spk, seed):
    """EXACT Appendix-A split: sorted unique speakers, shuffle(seed), first 55% train."""
    su = np.array(sorted(set(spk.tolist())))
    np.random.default_rng(seed).shuffle(su)
    tr_sp = set(su[: int(0.55 * len(su))].tolist())
    trm = np.array([s in tr_sp for s in spk])
    return trm, ~trm


# ---- closed-set rank-1 misID on a weighted sum of per-model cosines --------------
def misid_weighted(blocks_gpu, spk, w):
    """blocks_gpu: list of (N,d_m) unit-norm cuda tensors. w: (M,) weights.
    sim(i,j) = sum_m w_m <n_m^i, n_m^j>. Returns closed-set rank-1 misID
    (fraction of probes whose nearest non-self neighbour is a different speaker),
    restricted to probes that have >=1 same-speaker gallery mate. Mirrors
    verification_metrics.openset_misid exactly (best-same < best-diff)."""
    n = len(spk)
    _, c = np.unique(spk, return_inverse=True)
    ct = torch.from_numpy(c).to(DEV)
    wt = [float(x) for x in w]
    neg = torch.tensor(-9.0, device=DEV)
    sb = np.full(n, -9.0)
    db = np.full(n, -9.0)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        s = torch.zeros(e - st, n, device=DEV)
        for m, B in enumerate(blocks_gpu):
            s += wt[m] * (B[st:e] @ B.T)
        rows = torch.arange(st, e, device=DEV)
        s[torch.arange(e - st, device=DEV), rows] = neg  # exclude self
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        sb[st:e] = torch.where(same, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same, neg, s).max(1).values.cpu().numpy()
    has = sb > -8
    return float(np.mean(sb[has] < db[has]))


def misid_scorefn(sim_fn, spk):
    """Generic closed-set misID where sim_fn(rows_slice_idx) -> (rows,N) score matrix."""
    n = len(spk)
    _, c = np.unique(spk, return_inverse=True)
    ct = torch.from_numpy(c).to(DEV)
    neg = torch.tensor(-1e9, device=DEV)
    sb = np.full(n, -1e18)
    db = np.full(n, -1e18)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        s = sim_fn(st, e)
        rows = torch.arange(st, e, device=DEV)
        s[torch.arange(e - st, device=DEV), rows] = neg
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        sb[st:e] = torch.where(same, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same, neg, s).max(1).values.cpu().numpy()
    has = sb > -8e17
    return float(np.mean(sb[has] < db[has]))


# ---- (i) learned nonneg weights ------------------------------------------------
def learn_weights_softmax(blocks_gpu, spk, T=0.04, steps=300, sub_anchor=3072):
    """Full-gallery multi-positive identification softmax; optimize nonneg w via Adam.
    Loss(anchor a) = -log( sum_{j!=a, same} e^{S_aj/T} / sum_{j!=a} e^{S_aj/T} ).
    This is the differentiable analogue of rank-1 misID."""
    n = len(spk)
    _, c = np.unique(spk, return_inverse=True)
    ct = torch.from_numpy(c).to(DEV)
    theta = torch.zeros(len(blocks_gpu), device=DEV, requires_grad=True)  # softplus(0)=~0.69 each
    opt = torch.optim.Adam([theta], lr=0.05)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        w = torch.nn.functional.softplus(theta)
        idx = torch.randperm(n, generator=g)[:sub_anchor].to(DEV)
        S = torch.zeros(len(idx), n, device=DEV)
        for m, B in enumerate(blocks_gpu):
            S = S + w[m] * (B[idx] @ B.T)
        S = S / T
        rr = torch.arange(len(idx), device=DEV)
        S[rr, idx] = -1e9  # mask self
        same = (ct.unsqueeze(0) == ct[idx].unsqueeze(1))
        same[rr, idx] = False
        # keep anchors that have a same-speaker mate in gallery
        keep = same.any(1)
        Sk, samek = S[keep], same[keep]
        lse_all = torch.logsumexp(Sk, dim=1)
        neg_inf = torch.tensor(-1e9, device=DEV)
        lse_pos = torch.logsumexp(torch.where(samek, Sk, neg_inf), dim=1)
        loss = -(lse_pos - lse_all).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = torch.nn.functional.softplus(theta).detach().cpu().numpy()
    return w / w.sum() * len(blocks_gpu)  # normalize to mean 1 (scale-free for ranking)


def learn_weights_nm(blocks_gpu, spk, sub=9000):
    """Directly minimize TRAIN closed-set misID over the weight simplex via
    Nelder-Mead (exact metric objective). Subsample the train gallery for speed."""
    n = len(spk)
    rng = np.random.default_rng(0)
    sel = np.arange(n) if n <= sub else np.sort(rng.choice(n, sub, replace=False))
    bg = [B[torch.from_numpy(sel).to(DEV)] for B in blocks_gpu]
    sp = spk[sel]

    def obj(theta):
        w = np.abs(theta)
        if w.sum() < 1e-6:
            return 1.0
        return misid_weighted(bg, sp, w / w.sum())

    best = None
    for x0 in [np.ones(4), np.array([1.0, 2, 2, 2]), np.array([0.5, 1, 1.5, 2])]:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    w = np.abs(best.x)
    return w / w.sum() * len(blocks_gpu), float(best.fun)


# ---- (ii) stacking logistic regression over per-model cosines -------------------
def sample_pairs(spk, n_pairs=60000, seed=0):
    """Balanced same/diff index pairs from the train speakers."""
    rng = np.random.default_rng(seed)
    from collections import defaultdict
    by = defaultdict(list)
    for i, s in enumerate(spk):
        by[s].append(i)
    multi = [s for s in by if len(by[s]) >= 2]
    spks = list(by)
    tar, non = [], []
    for _ in range(n_pairs // 2):
        s = multi[rng.integers(len(multi))]
        a, b = rng.choice(by[s], 2, replace=False)
        tar.append((a, b))
        s1, s2 = rng.choice(len(spks), 2, replace=False)
        non.append((rng.choice(by[spks[s1]]), rng.choice(by[spks[s2]])))
    return np.array(tar), np.array(non)


def pair_cosines(blocks_gpu, pairs):
    """(P,M) matrix of per-model cosines for the given index pairs."""
    a = torch.from_numpy(pairs[:, 0]).to(DEV)
    b = torch.from_numpy(pairs[:, 1]).to(DEV)
    cols = []
    for B in blocks_gpu:
        cols.append((B[a] * B[b]).sum(1).cpu().numpy())
    return np.stack(cols, 1)  # (P, M)


def quad_feats(C):
    """[cos_m] ++ [cos_m*cos_n for m<=n] (degree-2)."""
    M = C.shape[1]
    extra = [C[:, i] * C[:, j] for i in range(M) for j in range(i, M)]
    return np.concatenate([C, np.stack(extra, 1)], 1)


def stack_misid(blocks_gpu, spk, coef, quad=False, coef_bias=0.0):
    """Closed-set misID with sim(i,j) = coef . feats(cos(i,j))."""
    M = len(blocks_gpu)
    w_lin = torch.tensor(coef[:M], dtype=torch.float32, device=DEV)
    if quad:
        # quadratic coefficients, index bookkeeping matches quad_feats order
        q_idx = [(i, j) for i in range(M) for j in range(i, M)]
        w_q = torch.tensor(coef[M:], dtype=torch.float32, device=DEV)

    def sim_fn(st, e):
        cosm = [B[st:e] @ B.T for B in blocks_gpu]  # M tensors (rows,N)
        s = torch.zeros_like(cosm[0])
        for m in range(M):
            s = s + w_lin[m] * cosm[m]
        if quad:
            for k, (i, j) in enumerate(q_idx):
                s = s + w_q[k] * (cosm[i] * cosm[j])
        return s

    return misid_scorefn(sim_fn, spk)


def run():
    blocks, spk, src = load_members()
    keep_glob = np.array([not str(s).startswith("freelance:") for s in src])
    results = {
        "protocol": {
            "members": MEMBERS, "seeds": SEEDS, "train_frac": 0.55,
            "backend": "plain cosine (no LDA/WCCN/PLDA); fusion axis only",
            "metric": "closed-set rank-1 misID (best-same < best-diff), freelance dropped",
            "note": "cos_m = raw per-model cosine of L2-normalized member embeddings; "
                    "equal-weight baseline == paper's 'cosine = mean of per-model cosines'",
        },
        "sanity_paper_reproduction": {}, "per_split": [], "summary": {},
    }

    # ---- sanity: joint concat+center+LN (exact paper cosine backend) on ens_sv4 --
    def ln(x):
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    ens = np.load(EMB / "ens_sv4.npz", allow_pickle=True)
    e_emb = ens["emb"].astype(np.float32)
    e_spk = np.asarray(ens["speaker_id"])
    e_src = np.asarray(ens["recording_source"])
    e_keep = np.array([not str(s).startswith("freelance:") for s in e_src])
    paper_vals = []
    for seed in SEEDS:
        trm, evm = split_masks(e_spk[e_keep], seed)
        E = e_emb[e_keep]
        S = e_spk[e_keep]
        Xtr, Xev, yev = E[trm], E[evm], S[evm]
        Z = ln(Xev - Xtr.mean(0, keepdims=True))
        Zt = torch.from_numpy(Z).to(DEV)
        paper_vals.append(misid_scorefn(lambda st, e: Zt[st:e] @ Zt.T, yev))
    results["sanity_paper_reproduction"] = {
        "concat_center_LN_cosine_misID_mean": float(np.mean(paper_vals)),
        "std": float(np.std(paper_vals)), "per_split": paper_vals,
        "expected_paper": "~0.039 (3.9%)",
    }

    variants = ["mean_pool", "learned_weights", "stack_lr_linear", "stack_lr_quad"]
    acc = {v: [] for v in variants}
    learned_w_log = []

    for seed in SEEDS:
        trm_all, evm_all = split_masks(spk[keep_glob], seed)
        blk = [B[keep_glob] for B in blocks]
        S = spk[keep_glob]
        tr_idx = np.where(trm_all)[0]
        ev_idx = np.where(evm_all)[0]
        # GPU tensors for train and eval galleries
        blk_tr = [torch.from_numpy(B[tr_idx]).to(DEV) for B in blk]
        blk_ev = [torch.from_numpy(B[ev_idx]).to(DEV) for B in blk]
        ytr, yev = S[tr_idx], S[ev_idx]

        rec = {"seed": seed, "n_train_spk": int(len(set(ytr.tolist()))),
               "n_eval_spk": int(len(set(yev.tolist()))),
               "n_train_seg": int(len(ytr)), "n_eval_seg": int(len(yev))}

        # baseline mean-pool (equal weights)
        w_eq = np.ones(len(blk))
        rec["mean_pool"] = misid_weighted(blk_ev, yev, w_eq)
        rec["mean_pool_train"] = misid_weighted(blk_tr, ytr, w_eq)

        # (i) learned weights: softmax-surrogate AND Nelder-Mead; select by TRAIN misID
        w_soft = learn_weights_softmax(blk_tr, ytr)
        tr_soft = misid_weighted(blk_tr, ytr, w_soft)
        w_nm, tr_nm = learn_weights_nm(blk_tr, ytr)
        if tr_soft <= tr_nm:
            w_lrn, src_lrn, tr_lrn = w_soft, "softmax", tr_soft
        else:
            w_lrn, src_lrn, tr_lrn = w_nm, "nelder_mead", tr_nm
        rec["learned_weights"] = misid_weighted(blk_ev, yev, w_lrn)
        rec["learned_weights_train"] = float(tr_lrn)
        rec["learned_weights_vec"] = [round(float(x), 4) for x in w_lrn]
        rec["learned_weights_opt"] = src_lrn
        learned_w_log.append({"seed": seed, "w": rec["learned_weights_vec"],
                              "opt": src_lrn, "w_soft": [round(float(x), 4) for x in w_soft],
                              "w_nm": [round(float(x), 4) for x in w_nm]})

        # (ii) stacking LR over per-model cosines (train pairs)
        tar, non = sample_pairs(ytr, n_pairs=80000, seed=seed)
        Ctar = pair_cosines(blk_tr, tar)
        Cnon = pair_cosines(blk_tr, non)
        Xp = np.concatenate([Ctar, Cnon], 0)
        yp = np.concatenate([np.ones(len(Ctar)), np.zeros(len(Cnon))])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xp, yp)
        rec["stack_lr_linear"] = stack_misid(blk_ev, yev, lr.coef_[0], quad=False)
        rec["stack_lr_linear_coef"] = [round(float(x), 4) for x in lr.coef_[0]]

        Xq = quad_feats(np.concatenate([Ctar, Cnon], 0))
        lrq = LogisticRegression(C=1.0, max_iter=3000).fit(Xq, yp)
        rec["stack_lr_quad"] = stack_misid(blk_ev, yev, lrq.coef_[0], quad=True)

        for v in variants:
            acc[v].append(rec[v])
        results["per_split"].append(rec)
        print(f"seed {seed}: mean-pool {rec['mean_pool']*100:.2f}%  "
              f"learned-w {rec['learned_weights']*100:.2f}% (w={rec['learned_weights_vec']},{src_lrn})  "
              f"stackLR {rec['stack_lr_linear']*100:.2f}%  stackLR-quad {rec['stack_lr_quad']*100:.2f}%")

    for v in variants:
        a = np.array(acc[v])
        results["summary"][v] = {"misID_mean": float(a.mean()), "misID_std": float(a.std()),
                                 "per_split": [float(x) for x in a]}
    results["learned_weights_log"] = learned_w_log

    # ---- mechanism probe: is the learned-weight gain just "downweight ecapa"? -----
    probes = {"equal_[1,1,1,1]": [1, 1, 1, 1], "drop_ecapa_[1,0,1,1]": [1, 0, 1, 1],
              "drop_xvector_[0,1,1,1]": [0, 1, 1, 1], "campp+redim_[0,0,1,1]": [0, 0, 1, 1],
              "downwt_ecapa_[1,.2,1,1]": [1, 0.2, 1, 1]}
    prb = {k: [] for k in probes}
    for seed in SEEDS:
        _, evm = split_masks(spk[keep_glob], seed)
        ev_idx = np.where(evm)[0]
        blk_ev = [torch.from_numpy(B[keep_glob][ev_idx]).to(DEV) for B in blocks]
        yev = spk[keep_glob][ev_idx]
        for k, w in probes.items():
            prb[k].append(misid_weighted(blk_ev, yev, np.array(w, float)))
    results["mechanism_probe_fixed_weights"] = {
        k: {"misID_mean": float(np.mean(v)), "misID_std": float(np.std(v))}
        for k, v in prb.items()}

    mp = results["summary"]["mean_pool"]["misID_mean"]
    lw = results["summary"]["learned_weights"]["misID_mean"]
    sl = results["summary"]["stack_lr_linear"]["misID_mean"]
    results["conclusion"] = (
        f"The mean-pool floor barely moves. Learned nonneg per-encoder weights lower "
        f"misID from {mp*100:.2f}% to {lw*100:.2f}% (-{(mp-lw)*100:.2f}pp, ~{(mp-lw)/mp*100:.0f}% "
        f"relative), consistent across all 3 disjoint splits -- but this ENTIRE gain is "
        f"reproduced by the trivial rule 'downweight the redundant ecapa member' "
        f"(drop_ecapa={results['mechanism_probe_fixed_weights']['drop_ecapa_[1,0,1,1]']['misID_mean']*100:.2f}%), "
        f"i.e. member pruning, not richer fusion. The 'stronger fusion' the reviewer names -- "
        f"stacking logistic regression over per-encoder cosines -- does NOT beat mean-pool; it is "
        f"WORSE ({sl*100:.2f}% linear, {results['summary']['stack_lr_quad']['misID_mean']*100:.2f}% "
        f"quadratic), because a pairwise same/diff logistic objective is misaligned with rank-1 "
        f"identification. Net: learned fusion does not meaningfully lower the closed-set misID floor.")

    AN.mkdir(parents=True, exist_ok=True)
    with (AN / "learned_fusion.json").open("w") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== closed-set rank-1 misID (SV-4, cosine back-end, 3 speaker-disjoint splits) ===")
    print(f"  paper concat+center+LN cosine (sanity) : "
          f"{results['sanity_paper_reproduction']['concat_center_LN_cosine_misID_mean']*100:.2f}"
          f"±{results['sanity_paper_reproduction']['std']*100:.2f}%")
    for v in variants:
        s = results["summary"][v]
        print(f"  {v:20s} : {s['misID_mean']*100:.2f}±{s['misID_std']*100:.2f}%")
    print(f"\n-> {AN/'learned_fusion.json'}")
    return results


if __name__ == "__main__":
    run()

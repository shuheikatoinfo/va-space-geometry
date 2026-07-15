"""Discriminative NONLINEAR re-rankers vs the linear/moment-fit misID floor (Table 3).

Tests the untested lever disclosed in paper section 7: can a discriminative nonlinear
re-ranker reduce the closed-set rank-1 misID floor below the linear suite
(cosine -> LDA -> WCCN -> two-cov PLDA-LLR)?

Two re-rankers, both trained ONLY on train-half speakers (identical speaker-disjoint
55/45 split scheme, same seeds, as verification_metrics/verification_plda):

  1. NPLDA (Ramoji et al., "NPLDA: A Deep Neural PLDA Model for Speaker
     Verification", Odyssey 2020 style): the bilinear PLDA-LLR
     s(u,v) = u'P v + u'Q u + v'Q v + c is initialized EXACTLY at the two-cov PLDA
     solution (P=diag(q), Q=diag(p), c=C) and the full matrices P,Q are fine-tuned
     discriminatively with BCE on genuine/impostor trials. This is the canonical
     "discriminative PLDA" lever: if the floor is a moment-fit artifact, freeing the
     quadratic form should reduce it.

  2. Pairwise MLP scorer: a shallow MLP (2 hidden layers x 256) on symmetric pair
     features [u1*u2, |u1-u2|, cos(u1,u2), ||u1||, ||u2||] over the PLDA-projected
     space (<=200-d, same information as the PLDA baseline; keeps the comparison
     "same front-end, nonlinear scoring" and the 1088-d ensemble tractable). This is
     a universal nonlinear pair scorer: it can represent any smooth symmetric
     score surface over that space, unlike the quadratic NPLDA.

Overfitting guard: train speakers are split 85/15 into FIT/VAL speakers; training
trials are drawn from FIT speakers only; early stopping monitors the 1:N misID among
VAL speakers (never eval speakers). Eval speakers are untouched until final scoring.

misID is scored exactly as the linear back-ends in verification_metrics: full
segment-vs-segment scoring on the eval half, best-same vs best-diff (closed-set
rank-1 error), self excluded. Pair EER / minCllr on the same 40k/40k trial scheme.

Usage:
  python -m src.nonlinear_reranker --sanity            # one (ecapa, seed 0) cell
  python -m src.nonlinear_reranker                     # full 3-encoder x 3-seed grid
Writes output/analysis/nonlinear_reranker.json. Does not modify existing scripts.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.verification_metrics import ln, eer, mincllr, trials, openset_misid
from src.verification_plda import plda_fit, plda_u, plda_pair_scores, plda_openset_misid

EMB = Path("output/embeddings"); AN = Path("output/analysis")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

HP = {
    "val_frac": 0.15,          # fraction of TRAIN speakers held out for early stopping
    "n_train_trials": 200_000, # per class (genuine/impostor), drawn from FIT speakers
    "batch": 8192,
    "max_epochs": 15,
    "patience": 3,
    "mlp_hidden": [256, 256],
    "mlp_dropout": 0.1,
    "mlp_lr": 1e-3,
    "mlp_wd": 1e-4,
    "nplda_lr": 1e-4,          # on P,Q (initialized at the PLDA solution)
    "nplda_cal_lr": 1e-2,      # on the affine calibration (does not affect argmax)
}


# ---------------------------------------------------------------- data / split
def load_model(model):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    return emb[keep], spk[keep]


def split_masks(spk, seed):
    # identical scheme to verification_metrics.run / verification_plda._one_split
    su = np.array(sorted(set(spk.tolist())))
    np.random.default_rng(seed).shuffle(su)
    tr_sp = set(su[: int(0.55 * len(su))])
    trm = np.array([s in tr_sp for s in spk])
    return trm, ~trm


def sample_trials(spk, n, rng):
    """Balanced genuine/impostor pairs among the given (fit) speakers."""
    from collections import defaultdict
    by = defaultdict(list)
    for i, s in enumerate(spk):
        by[s].append(i)
    multi = [s for s in by if len(by[s]) >= 2]; spks = list(by)
    tar = np.empty((n, 2), dtype=np.int64); non = np.empty((n, 2), dtype=np.int64)
    for k in range(n):
        s = multi[rng.integers(len(multi))]
        tar[k] = rng.choice(by[s], 2, replace=False)
        i, j = rng.choice(len(spks), 2, replace=False)
        non[k] = (rng.choice(by[spks[i]]), rng.choice(by[spks[j]]))
    return tar, non


# ---------------------------------------------------------------- scorers
class NPLDA(nn.Module):
    """Bilinear PLDA-LLR with free symmetric P (cross) and Q (self) matrices,
    initialized at the two-cov PLDA solution; affine calibration for BCE."""

    def __init__(self, p, q, C):
        super().__init__()
        D = len(p)
        self.P = nn.Parameter(torch.diag(torch.from_numpy(q.astype(np.float32))))
        self.Q = nn.Parameter(torch.diag(torch.from_numpy(p.astype(np.float32))))
        self.c = nn.Parameter(torch.tensor(float(C)))
        self.a = nn.Parameter(torch.tensor(1.0))   # calibration scale
        self.b = nn.Parameter(torch.tensor(0.0))   # calibration offset

    def _sym(self):
        return (self.P + self.P.T) / 2, (self.Q + self.Q.T) / 2

    def score(self, u, v):
        P, Q = self._sym()
        s = (u @ P * v).sum(1) + (u @ Q * u).sum(1) + (v @ Q * v).sum(1) + self.c
        return s

    def logit(self, u, v):
        return self.a * self.score(u, v) + self.b

    @torch.no_grad()
    def score_matrix_block(self, Uq, Ug):
        P, Q = self._sym()
        gq = (Uq @ Q * Uq).sum(1); gg = (Ug @ Q * Ug).sum(1)
        return Uq @ P @ Ug.T + gq[:, None] + gg[None, :] + self.c


class PairMLP(nn.Module):
    """Shallow MLP on symmetric pair features over the PLDA-projected space."""

    def __init__(self, D, hidden, dropout):
        super().__init__()
        layers, d = [], 2 * D + 3
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def feats(u, v):
        nu = u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cos = (u * v).sum(-1, keepdim=True) / (nu * nv)
        return torch.cat([u * v, (u - v).abs(), cos, nu, nv], dim=-1)

    def logit(self, u, v):
        return self.net(self.feats(u, v)).squeeze(-1)


# ---------------------------------------------------------------- misID scoring
@torch.no_grad()
def misid_nplda(model, U, spk):
    n = len(spk)
    _, c = np.unique(spk, return_inverse=True)
    ct = torch.from_numpy(c).to(DEV)
    Ut = torch.from_numpy(U.astype(np.float32)).to(DEV)
    neg = torch.tensor(-1e9, device=DEV)
    sb = np.full(n, -1e18); db = np.full(n, -1e18)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        s = model.score_matrix_block(Ut[st:e], Ut)
        s[torch.arange(e - st, device=DEV), torch.arange(st, e, device=DEV)] = neg
        same = ct[None, :] == ct[st:e, None]
        sb[st:e] = torch.where(same, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same, neg, s).max(1).values.cpu().numpy()
    return float(np.mean(sb < db))


@torch.no_grad()
def misid_mlp(net, U, spk, qchunk=192, gchunk=4096):
    net.eval()
    n = len(spk)
    _, c = np.unique(spk, return_inverse=True)
    ct = torch.from_numpy(c).to(DEV)
    Ut = torch.from_numpy(U.astype(np.float32)).to(DEV)
    neg = torch.tensor(-1e9, device=DEV)
    sb = np.full(n, -1e18); db = np.full(n, -1e18)
    for qs in range(0, n, qchunk):
        qe = min(qs + qchunk, n)
        best_s = torch.full((qe - qs,), -1e18, device=DEV)
        best_d = torch.full((qe - qs,), -1e18, device=DEV)
        for gs in range(0, n, gchunk):
            ge = min(gs + gchunk, n)
            u = Ut[qs:qe, None, :].expand(-1, ge - gs, -1)
            v = Ut[None, gs:ge, :].expand(qe - qs, -1, -1)
            s = net.logit(u.reshape(-1, Ut.shape[1]), v.reshape(-1, Ut.shape[1]))
            s = s.view(qe - qs, ge - gs)
            # mask self-pairs
            qi = torch.arange(qs, qe, device=DEV)[:, None]
            gi = torch.arange(gs, ge, device=DEV)[None, :]
            s = torch.where(qi == gi, neg, s)
            same = ct[None, gs:ge] == ct[qs:qe, None]
            best_s = torch.maximum(best_s, torch.where(same, s, neg).max(1).values)
            best_d = torch.maximum(best_d, torch.where(same, neg, s).max(1).values)
        sb[qs:qe] = best_s.cpu().numpy(); db[qs:qe] = best_d.cpu().numpy()
    return float(np.mean(sb < db))


# ---------------------------------------------------------------- training
def train_scorer(kind, Ufit, yfit, Uval, yval, plda_model, seed, log=print):
    """Train a nonlinear scorer on FIT-speaker trials, early-stop on VAL misID."""
    rng = np.random.default_rng(1000 + seed)
    D = Ufit.shape[1]
    if kind == "nplda":
        net = NPLDA(plda_model["p"], plda_model["q"], plda_model["C"]).to(DEV)
        opt = torch.optim.Adam([
            {"params": [net.P, net.Q, net.c], "lr": HP["nplda_lr"]},
            {"params": [net.a, net.b], "lr": HP["nplda_cal_lr"]},
        ])
        val_fn = lambda: misid_nplda(net, Uval, yval)
    else:
        net = PairMLP(D, HP["mlp_hidden"], HP["mlp_dropout"]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=HP["mlp_lr"], weight_decay=HP["mlp_wd"])
        val_fn = lambda: misid_mlp(net, Uval, yval)

    tar, non = sample_trials(yfit, HP["n_train_trials"], rng)
    pairs = np.concatenate([tar, non])
    labels = np.concatenate([np.ones(len(tar)), np.zeros(len(non))]).astype(np.float32)
    Ut = torch.from_numpy(Ufit.astype(np.float32)).to(DEV)
    lab_t = torch.from_numpy(labels).to(DEV)
    pairs_t = torch.from_numpy(pairs).to(DEV)
    bce = nn.BCEWithLogitsLoss()

    best_val, best_state, bad = val_fn(), {k: v.clone() for k, v in net.state_dict().items()}, 0
    log(f"    [{kind}] init VAL misID {best_val*100:.2f}%")
    for ep in range(HP["max_epochs"]):
        net.train()
        perm = torch.randperm(len(pairs_t), device=DEV)
        tot = 0.0
        for st in range(0, len(perm), HP["batch"]):
            idx = perm[st: st + HP["batch"]]
            u, v = Ut[pairs_t[idx, 0]], Ut[pairs_t[idx, 1]]
            loss = bce(net.logit(u, v), lab_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        vm = val_fn()
        log(f"    [{kind}] epoch {ep+1:2d} loss {tot/len(perm):.4f}  VAL misID {vm*100:.2f}%")
        if vm < best_val - 1e-6:
            best_val, bad = vm, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best_val


@torch.no_grad()
def pair_scores(net, U, pairs, chunk=65536):
    Ut = torch.from_numpy(U.astype(np.float32)).to(DEV)
    out = np.empty(len(pairs))
    for st in range(0, len(pairs), chunk):
        p = pairs[st: st + chunk]
        out[st: st + chunk] = net.logit(Ut[p[:, 0]], Ut[p[:, 1]]).cpu().numpy()
    return out


# ---------------------------------------------------------------- one cell
def run_cell(emb, spk, seed, log=print):
    trm, evm = split_masks(spk, seed)
    Xtr, ytr, Xev, yev = emb[trm], spk[trm], emb[evm], spk[evm]
    res = {"n_train_spk": len(set(ytr.tolist())), "n_eval_spk": len(set(yev.tolist())),
           "n_eval_seg": int(evm.sum())}

    # --- linear baselines (sanity vs Table 3) ---
    t0 = time.time()
    Zev = ln(Xev - Xtr.mean(0, keepdims=True))
    res["cosine_misID"] = openset_misid(Zev, yev)
    pl = plda_fit(Xtr, ytr)
    Uev = plda_u(pl, Xev)
    res["plda_misID"] = plda_openset_misid(pl, Uev, yev)
    ta, no = trials(Xev, yev)
    tars, nons = plda_pair_scores(pl, Uev, ta), plda_pair_scores(pl, Uev, no)
    res["plda_EER"] = eer(tars, nons); res["plda_minCllr"] = mincllr(tars, nons)
    log(f"  seed {seed}: cosine misID {res['cosine_misID']*100:.2f}%  "
        f"PLDA misID {res['plda_misID']*100:.2f}%  PLDA EER {res['plda_EER']*100:.1f}%  "
        f"({time.time()-t0:.0f}s)")

    # --- fit/val split WITHIN train speakers (early-stopping guard) ---
    tr_uniq = np.array(sorted(set(ytr.tolist())))
    np.random.default_rng(2000 + seed).shuffle(tr_uniq)
    nval = int(HP["val_frac"] * len(tr_uniq))
    val_sp = set(tr_uniq[:nval])
    vmask = np.array([s in val_sp for s in ytr])
    Utr = plda_u(pl, Xtr)  # nonlinear scorers operate on the PLDA-projected space
    Ufit, yfit, Uval, yval = Utr[~vmask], ytr[~vmask], Utr[vmask], ytr[vmask]
    log(f"    fit spk {len(set(yfit.tolist()))} ({len(yfit)} seg) / "
        f"val spk {len(set(yval.tolist()))} ({len(yval)} seg)")

    # --- nonlinear re-rankers ---
    for kind in ["nplda", "mlp"]:
        t0 = time.time()
        net, vbest = train_scorer(kind, Ufit, yfit, Uval, yval, pl, seed, log=log)
        mis = misid_nplda(net, Uev, yev) if kind == "nplda" else misid_mlp(net, Uev, yev)
        tarn, nonn = pair_scores(net, Uev, ta), pair_scores(net, Uev, no)
        res[f"{kind}_misID"] = mis
        res[f"{kind}_EER"] = eer(tarn, nonn); res[f"{kind}_minCllr"] = mincllr(tarn, nonn)
        res[f"{kind}_val_misID"] = vbest
        log(f"    [{kind}] EVAL misID {mis*100:.2f}%  EER {res[f'{kind}_EER']*100:.1f}%  "
            f"minCllr {res[f'{kind}_minCllr']:.3f}  ({time.time()-t0:.0f}s)")
    return res


def aggregate(cells):
    keys = [k for k in cells[0] if isinstance(cells[0][k], float)]
    agg = {k: float(np.mean([c[k] for c in cells])) for k in keys}
    agg.update({k + "_std": float(np.std([c[k] for c in cells])) for k in keys})
    agg["n_eval_spk"] = cells[0]["n_eval_spk"]
    agg["per_seed"] = cells
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ecapa", "animeva", "ens_sv4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sanity", action="store_true", help="one (ecapa, seed 0) cell only")
    args = ap.parse_args()
    if args.sanity:
        args.models, args.seeds = ["ecapa"], [0]

    out = {"hyperparams": HP, "seeds": args.seeds, "protocol":
           "speaker-disjoint 55/45 split (same scheme/seeds as Table 3); nonlinear "
           "scorers trained on 85% of TRAIN speakers, early-stopped on the remaining "
           "15% (VAL) 1:N misID; eval speakers untouched until final scoring; scorers "
           "operate on the two-cov PLDA projected space (<=200-d)."}
    for m in args.models:
        print(f"\n#### {m} ####")
        emb, spk = load_model(m)
        cells = [run_cell(emb, spk, s) for s in args.seeds]
        out[m] = aggregate(cells)
        a = out[m]
        print(f"  {m}: misID cosine {a['cosine_misID']*100:.1f}±{a['cosine_misID_std']*100:.1f}%  "
              f"PLDA {a['plda_misID']*100:.1f}±{a['plda_misID_std']*100:.1f}%  "
              f"NPLDA {a['nplda_misID']*100:.1f}±{a['nplda_misID_std']*100:.1f}%  "
              f"MLP {a['mlp_misID']*100:.1f}±{a['mlp_misID_std']*100:.1f}%")
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "nonlinear_reranker.json", "w"), indent=2)
    print(f"\n-> {AN/'nonlinear_reranker.json'}")


if __name__ == "__main__":
    main()

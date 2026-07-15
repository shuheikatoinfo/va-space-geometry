"""Session-disjoint closed-set misID under the full discriminative re-ranking suite
(cosine -> LDA -> WCCN -> two-covariance PLDA-LLR).

Reviewer must-fix: verification_metrics.py / verification_plda.py evaluate the
closed-set rank-1 misID with SAME-SESSION genuine comparisons allowed (a query's
best same-speaker match may come from the same source file). Here the protocol is
identical -- same speaker-disjoint 55/45 splits (seeds 0..n_splits-1), back-ends
trained identically on the train half -- except that at evaluation time the
genuine side is constrained to be SESSION-DISJOINT: S_same for a query is the max
score over the query speaker's own segments from a DIFFERENT source file
(source_sha256, joined to the npz via segment_id from segments.jsonl; the npz
`recording_source` is agency-level and cannot define sessions). S_diff is
unchanged (max over all other speakers' segments). Queries whose speaker has
segments from only one source file are dropped and counted.

Does NOT modify verification_plda.py / verification_metrics.py; imports from them.

Usage: python -m src.session_disjoint_backend --models ens_sv4 animeva ecapa
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from src.verification_metrics import backend_transform
from src.verification_plda import plda_fit, plda_u

EMB = Path("output/embeddings")
AN = Path("output/analysis")
SEGMENTS = Path("data/processed/segments.jsonl")

BACKENDS = ["cosine", "lda", "wccn", "plda"]


def segid_to_sha():
    """segment_id -> source_sha256 (deduped, as in centered_gallery.py)."""
    mp = {}
    for l in open(SEGMENTS, encoding="utf-8"):
        if '"segment_id"' not in l:
            continue
        r = json.loads(l)
        mp.setdefault(r["segment_id"], r["source_sha256"])
    return mp


def _pairwise_misid(score_rows, spk_codes, sha_codes):
    """Generic closed-set misID with session-disjoint genuine constraint.

    score_rows(st, e) must return the (e-st) x N score block with self-scores
    already valid (they get masked here). Returns (misid, n_used, n_dropped)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ct = torch.from_numpy(spk_codes).to(dev)
    ht = torch.from_numpy(sha_codes).to(dev)
    n = len(spk_codes)
    neg = torch.tensor(-1e18, device=dev)
    sb = np.full(n, -1e18)
    db = np.full(n, -1e18)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        s = score_rows(st, e).to(dev)
        same_spk = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        same_sha = ht.unsqueeze(0) == ht[st:e].unsqueeze(1)
        # genuine: same speaker AND different source file (also excludes self)
        gen = same_spk & ~same_sha
        sb[st:e] = torch.where(gen, s, neg).max(1).values.cpu().numpy()
        db[st:e] = torch.where(same_spk, neg, s).max(1).values.cpu().numpy()
    usable = sb > -1e17  # query has >=1 same-speaker segment from another file
    n_used = int(usable.sum())
    misid = float(np.mean(sb[usable] < db[usable]))
    return misid, n_used, n - n_used


def misid_cosine_space(Z, spk_codes, sha_codes):
    import torch
    Zt = torch.from_numpy(Z.astype(np.float32))
    if torch.cuda.is_available():
        Zt = Zt.cuda()
    return _pairwise_misid(lambda st, e: Zt[st:e] @ Zt.T, spk_codes, sha_codes)


def misid_plda(model, U, spk_codes, sha_codes):
    import torch
    p = torch.from_numpy(model["p"].astype(np.float32))
    q = torch.from_numpy(model["q"].astype(np.float32))
    Ut = torch.from_numpy(U.astype(np.float32))
    if torch.cuda.is_available():
        p, q, Ut = p.cuda(), q.cuda(), Ut.cuda()
    g = (Ut * Ut) @ p
    Uq = Ut * q
    C = model["C"]

    def rows(st, e):
        return Ut[st:e] @ Uq.T + g.unsqueeze(0) + g[st:e].unsqueeze(1) + C

    return _pairwise_misid(rows, spk_codes, sha_codes)


def one_split(emb, spk, sha, seed):
    """Mirror of the verification_metrics/verification_plda split scheme."""
    su = np.array(sorted(set(spk.tolist())))
    np.random.default_rng(seed).shuffle(su)
    tr_sp = set(su[: int(0.55 * len(su))])
    trm = np.array([s in tr_sp for s in spk]); evm = ~trm
    Xtr, ytr = emb[trm], spk[trm]
    Xev, yev, hev = emb[evm], spk[evm], sha[evm]
    _, spk_codes = np.unique(yev, return_inverse=True)
    _, sha_codes = np.unique(hev, return_inverse=True)
    res = {"n_eval_spk": int(len(set(yev.tolist()))), "n_eval_seg": int(len(yev))}
    for kind in ["cosine", "lda", "wccn"]:
        Z = backend_transform(Xtr, ytr, Xev, kind)
        m, nu, nd = misid_cosine_space(Z, spk_codes, sha_codes)
        res[kind] = {"misid": m, "n_query_used": nu, "n_query_dropped": nd}
    pl = plda_fit(Xtr, ytr)
    U = plda_u(pl, Xev)
    m, nu, nd = misid_plda(pl, U, spk_codes, sha_codes)
    res["plda"] = {"misid": m, "n_query_used": nu, "n_query_dropped": nd}
    # eval speakers with >=2 distinct source files (effective genuine population)
    multi = 0
    for s in set(yev.tolist()):
        if len(set(hev[yev == s].tolist())) >= 2:
            multi += 1
    res["n_eval_spk_multisession"] = multi
    return res


def run_model(model, n_splits):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    spk = np.asarray(d["speaker_id"])
    src = np.asarray(d["recording_source"])
    sid = d["segment_id"].astype(str)
    mp = segid_to_sha()
    sha = np.array([mp.get(s, f"__missing__{s}") for s in sid])
    n_missing = int(sum(1 for s in sid if s not in mp))
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    emb, spk, sha = emb[keep], spk[keep], sha[keep]
    splits = [one_split(emb, spk, sha, s) for s in range(n_splits)]
    agg = {"n_splits": n_splits, "n_missing_sha": n_missing, "splits": splits}
    for kind in BACKENDS:
        vals = [sp[kind]["misid"] for sp in splits]
        agg[kind] = {"misid": float(np.mean(vals)), "misid_std": float(np.std(vals)),
                     "n_query_used_mean": float(np.mean([sp[kind]["n_query_used"] for sp in splits])),
                     "n_query_dropped_mean": float(np.mean([sp[kind]["n_query_dropped"] for sp in splits]))}
    agg["n_eval_spk_mean"] = float(np.mean([sp["n_eval_spk"] for sp in splits]))
    agg["n_eval_spk_multisession_mean"] = float(np.mean([sp["n_eval_spk_multisession"] for sp in splits]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ens_sv4", "animeva", "ecapa"])
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args()
    out = {"_meta": {
        "protocol": ("Speaker-disjoint 55/45 splits (seeds 0..n_splits-1, identical scheme to "
                     "verification_metrics.py/verification_plda.py). Back-ends (cosine center+LN, "
                     "LDA-150+cos, WCCN+cos, two-cov PLDA-LLR) trained on the 55% train-half "
                     "speakers, evaluated on the 45% eval half (agency-only, freelance: dropped). "
                     "Closed-set rank-1 misID with SESSION-DISJOINT genuine trials: S_same = max "
                     "score over the query speaker's own segments from a DIFFERENT source file "
                     "(source_sha256 via segments.jsonl segment_id join); S_diff = max over other "
                     "speakers' segments (unchanged). Queries whose eval-half speaker has segments "
                     "from only one source file are dropped (counted per back-end/split)."),
        "session_key": "source_sha256 (per source file), NOT recording_source (agency-level)",
        "misid_def": "fraction of usable queries with best different-speaker score > best session-disjoint same-speaker score",
    }, "results": {}}
    print(f"=== Session-disjoint closed-set misID under re-ranking ({args.n_splits} splits) ===")
    hdr = f"{'model':9s} {'Nspk':>5s} {'Nspk2+':>6s} {'Nquery':>7s} " + " ".join(f"{b:>12s}" for b in BACKENDS)
    print(hdr)
    for m in args.models:
        r = run_model(m, args.n_splits)
        out["results"][m] = r
        cells = " ".join(f"{r[b]['misid']*100:5.1f}±{r[b]['misid_std']*100:.1f}%" for b in BACKENDS)
        print(f"{m:9s} {r['n_eval_spk_mean']:5.0f} {r['n_eval_spk_multisession_mean']:6.0f} "
              f"{r['cosine']['n_query_used_mean']:7.0f} {cells}")
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "session_disjoint_backend.json", "w"), indent=2)
    print(f"-> {AN / 'session_disjoint_backend.json'}")


if __name__ == "__main__":
    main()

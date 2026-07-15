"""Control-group comparison through the FULL discriminative re-ranking suite
(cosine -> LDA -> WCCN -> two-covariance PLDA-LLR), i.e. the Table-3 backends,
instead of only the AS-norm stage used in §4.3.

For each control corpus (JVS neutral read, JVS-varied style condition, Common
Voice JA) and each representative encoder (ecapa, animeva, ens_sv4) we:
  1. match speaker count and segments-per-speaker exactly as in
     src.population_compare (§4.3 AS-norm comparison): the control's eligible
     speakers (>= m_seg segments) x m_seg segments, 3 seeds;
  2. draw a VA subset (agency-only, freelance excluded) matched to the SAME
     n_speakers x m_seg with the same seed, so the VA/control ratio is
     apples-to-apples (Table 3's N=497 numbers are NOT reused);
  3. run the identical backend protocol as Table 3: speaker-disjoint 55/45
     split (same seed), cosine / LDA(<=min(150, K_train-1)) / trace-relative
     ridge WCCN / PCA-pre-reduced moment-fit two-cov PLDA-LLR, reporting the
     closed-set rank-1 misID (JSON key `openset_misID` for artifact stability),
     pairwise EER and minCllr, averaged over the 3 seeds.

jxvector is skipped (ceiling-saturated per §4.3); diagnostic probes skipped.
Because CV has few speakers with >= 8 segments, a supplementary CV run at
m_seg=4 is included.

Requires control embeddings for the ens_sv4 members (xvector, ecapa, campp,
redimnet); builds output/embeddings_control/<corpus>_ens_sv4.npz on the fly if
missing (same mean-cosine fusion as src.make_ensemble).

Usage: python -m src.controls_plda [--n-seeds 3] [--m-seg 8]
Output: output/analysis/controls_plda.json
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.population_compare import load_pop, match_subset
from src.verification_metrics import backend_transform, eer, mincllr, openset_misid, trials
from src.verification_plda import plda_fit, plda_u, plda_pair_scores, plda_openset_misid

EMB = Path("output/embeddings")
CTRL = Path("output/embeddings_control")
AN = Path("output/analysis")
ENS_MEMBERS = ["xvector", "ecapa", "campp", "redimnet"]
BACKENDS = ["cosine", "lda", "wccn", "plda"]


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


def ensure_control_ensemble(corpus: str) -> Path:
    """Build <corpus>_ens_sv4.npz (concat of L2-normed members, aligned on segment_id)."""
    out = CTRL / f"{corpus}_ens_sv4.npz"
    if out.exists():
        return out
    loaded = {m: np.load(CTRL / f"{corpus}_{m}.npz", allow_pickle=True) for m in ENS_MEMBERS}
    common = set.intersection(*[set(d["segment_id"].tolist()) for d in loaded.values()])
    base = loaded[ENS_MEMBERS[0]]
    order = [s for s in base["segment_id"].tolist() if s in common]
    pos = {s: i for i, s in enumerate(order)}
    blocks = []
    for m in ENS_MEMBERS:
        d = loaded[m]
        idx = np.full(len(order), -1, np.int64)
        for i, s in enumerate(d["segment_id"].tolist()):
            if s in pos:
                idx[pos[s]] = i
        blocks.append(l2norm(d["emb"].astype(np.float32))[idx])
    fused = np.concatenate(blocks, 1)
    idx0 = {s: i for i, s in enumerate(base["segment_id"].tolist())}
    take = np.array([idx0[s] for s in order])
    np.savez(out, emb=fused, segment_id=np.array(order),
             speaker_id=base["speaker_id"][take], recording_source=base["recording_source"][take])
    print(f"built {out} {fused.shape}")
    return out


def eval_backends(emb, spk, seed):
    """Table-3 protocol on one matched subset: speaker-disjoint 55/45 split,
    all four backends. Returns metrics + dims actually used."""
    su = np.array(sorted(set(spk.tolist())))
    np.random.default_rng(seed).shuffle(su)
    tr_sp = set(su[: int(0.55 * len(su))])
    trm = np.array([s in tr_sp for s in spk]); evm = ~trm
    Xtr, ytr, Xev, yev = emb[trm], spk[trm], emb[evm], spk[evm]
    K = len(set(ytr.tolist()))
    ta, no = trials(Xev, yev)
    out = {"n_train_spk": K, "n_eval_spk": int(len(set(yev.tolist()))),
           "n_eval_seg": int(len(yev)), "n_tar_trials": int(len(ta)), "n_non_trials": int(len(no)),
           "lda_dim": int(min(150, K - 1)), "plda_pca_dim": int(min(200, K - 1, emb.shape[1]))}
    for kind in ["cosine", "lda", "wccn"]:
        Z = backend_transform(Xtr, ytr, Xev, kind)
        tar = np.sum(Z[ta[:, 0]] * Z[ta[:, 1]], 1); non = np.sum(Z[no[:, 0]] * Z[no[:, 1]], 1)
        out[kind] = {"EER": eer(tar, non), "minCllr": mincllr(tar, non),
                     "openset_misID": openset_misid(Z, yev)}
    pl = plda_fit(Xtr, ytr); U = plda_u(pl, Xev)
    tar = plda_pair_scores(pl, U, ta); non = plda_pair_scores(pl, U, no)
    out["plda"] = {"EER": eer(tar, non), "minCllr": mincllr(tar, non),
                   "openset_misID": plda_openset_misid(pl, U, yev)}
    return out


def agg_seeds(runs):
    out = {k: runs[0][k] for k in ["n_train_spk", "n_eval_spk", "n_eval_seg",
                                   "n_tar_trials", "n_non_trials", "lda_dim", "plda_pca_dim"]}
    for b in BACKENDS:
        out[b] = {}
        for metric in runs[0][b]:
            vals = [r[b][metric] for r in runs]
            out[b][metric] = float(np.mean(vals)); out[b][metric + "_std"] = float(np.std(vals))
    return out


def run_cell(model, corpus, m_seg, n_seeds):
    emb_c, spk_c = load_pop(model, corpus)
    emb_v, spk_v = load_pop(model, "va")
    by = defaultdict(int)
    for s in spk_c:
        by[s] += 1
    n_spk = sum(1 for v in by.values() if v >= m_seg)
    runs_c, runs_v = [], []
    for sd in range(n_seeds):
        ec, sc = match_subset(emb_c, spk_c, n_spk, m_seg, seed=sd)
        ev, sv = match_subset(emb_v, spk_v, n_spk, m_seg, seed=sd)
        runs_c.append(eval_backends(ec, sc, seed=sd))
        runs_v.append(eval_backends(ev, sv, seed=sd))
    cell = {"n_matched_spk": n_spk, "m_seg": m_seg, "n_seeds": n_seeds,
            "control": agg_seeds(runs_c), "va_matched": agg_seeds(runs_v)}
    cell["ratio"] = {}
    for b in BACKENDS:
        c, v = cell["control"][b]["openset_misID"], cell["va_matched"][b]["openset_misID"]
        cell["ratio"][b] = {"misID_va": v, "misID_control": c,
                            "va_over_control": (float(v / c) if c > 0 else None),
                            "control_misID_is_zero": bool(c == 0)}
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ecapa", "animeva", "ens_sv4"])
    ap.add_argument("--corpora", nargs="+", default=["jvs", "jvsv", "cv"])
    ap.add_argument("--m-seg", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--cv-supplementary-mseg", type=int, default=4,
                    help="extra CV run at this m_seg (CV has few speakers with >=8 segs); 0 disables")
    args = ap.parse_args()

    if "ens_sv4" in args.models:
        for c in args.corpora:
            ensure_control_ensemble(c)

    res = {"protocol": {"split": "speaker-disjoint 55/45", "backends": BACKENDS,
                        "matching": "population_compare.match_subset (control-eligible speakers x m_seg), VA agency-only",
                        "n_seeds": args.n_seeds, "m_seg": args.m_seg}}
    for model in args.models:
        res[model] = {}
        for corpus in args.corpora:
            print(f"\n=== {model} x {corpus} (m_seg={args.m_seg}) ===")
            cell = run_cell(model, corpus, args.m_seg, args.n_seeds)
            res[model][corpus] = cell
            _print_cell(cell)
            if corpus == "cv" and args.cv_supplementary_mseg:
                print(f"\n=== {model} x cv (supplementary m_seg={args.cv_supplementary_mseg}) ===")
                cell = run_cell(model, "cv", args.cv_supplementary_mseg, args.n_seeds)
                res[model]["cv_mseg%d" % args.cv_supplementary_mseg] = cell
                _print_cell(cell)
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(AN / "controls_plda.json", "w"), indent=2)
    print(f"\n-> {AN / 'controls_plda.json'}")


def _print_cell(cell):
    c, v = cell["control"], cell["va_matched"]
    print(f"  matched: {cell['n_matched_spk']} spk x {cell['m_seg']} segs | "
          f"train spk {c['n_train_spk']} | LDA dim {c['lda_dim']} | PLDA PCA dim {c['plda_pca_dim']} | "
          f"eval probes ctrl={c['n_eval_seg']} va={v['n_eval_seg']}")
    print(f"  {'backend':7s} {'ctrl misID%':>11s} {'VA misID%':>10s} {'ratio':>7s} {'ctrl EER%':>9s} {'VA EER%':>8s}")
    for b in BACKENDS:
        r = cell["ratio"][b]
        rat = f"{r['va_over_control']:.1f}x" if r["va_over_control"] is not None else "inf(0)"
        print(f"  {b:7s} {c[b]['openset_misID']*100:11.2f} {v[b]['openset_misID']*100:10.2f} {rat:>7s} "
              f"{c[b]['EER']*100:9.2f} {v[b]['EER']*100:8.2f}")


if __name__ == "__main__":
    main()

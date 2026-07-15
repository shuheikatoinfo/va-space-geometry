"""Reviewer-grade calibrated evaluation: does the thin-margin / hubness pathology
survive proper speaker-verification scoring (not just raw cosine)?

Backends:
  - raw     : cosine on raw embeddings (the naive baseline we are rebutting)
  - center  : cosine after mean-centering + length-norm
  - asnorm  : adaptive symmetric score normalization (AS-norm) over an impostor
              cohort, the field-standard remedy for non-transferable thresholds

For each backend and model (agency-only, codec-homogeneous subset) we report:
  - rank-1 identification accuracy
  - verification EER (from sampled same/different-speaker trials)
  - critical-margin fraction (|S_same - S_diff| < 0.02) with the margin computed
    on the *calibrated* scores
  - misidentification fraction (S_same < S_diff)
  - hubness skewness of the k-occurrence distribution
All point estimates carry a speaker-level bootstrap 95% CI (segments within a
speaker are correlated, so we resample speakers, not segments).

Usage:
    python -m src.calibrated_eval --models ecapa campp redimnet ens_sv4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

EMB_DIR = Path("output/embeddings")
SEGMENTS = Path("data/processed/segments.jsonl")
OUT = Path("output/analysis/calibrated.json")
SD_OUT = Path("output/analysis/session_disjoint.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def segid_to_sha():
    """segment_id -> source_sha256 (deduped, as in centered_gallery.py). Used to
    define per-file recording sessions; the npz `recording_source` is agency-level
    (1 per speaker) and cannot separate a speaker's own source files."""
    mp = {}
    for l in open(SEGMENTS, encoding="utf-8"):
        if '"segment_id"' not in l:
            continue
        r = json.loads(l)
        mp.setdefault(r["segment_id"], r["source_sha256"])
    return mp


def load_agency_only(model: str, with_sha: bool = False):
    d = np.load(EMB_DIR / f"{model}.npz", allow_pickle=True)
    spk = np.asarray(d["speaker_id"])
    src = np.asarray(d["recording_source"]) if "recording_source" in d else np.array(["?"] * len(spk))
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    emb = d["emb"][keep].astype(np.float32)
    if with_sha:
        sid = np.asarray(d["segment_id"]).astype(str)
        mp = segid_to_sha()
        sha = np.array([mp.get(s, f"__missing__{s}") for s in sid])[keep]
        return emb, spk[keep], sha
    return emb, spk[keep]


def preprocess(emb: np.ndarray, backend: str) -> np.ndarray:
    x = emb.copy()
    if backend in ("center", "asnorm"):
        x = x - x.mean(0, keepdims=True)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    return x


def cohort_stats(Xt: torch.Tensor, spk_t: torch.Tensor, cohort_n: int, chunk=2048):
    """Per-segment mean/std of its top-cohort_n impostor cosine similarities."""
    n = Xt.shape[0]
    mu = torch.empty(n, device=Xt.device)
    sd = torch.empty(n, device=Xt.device)
    neg = torch.tensor(-2.0, device=Xt.device)
    for st in range(0, n, chunk):
        e = min(st + chunk, n)
        sims = Xt[st:e] @ Xt.T
        same = spk_t.unsqueeze(0) == spk_t[st:e].unsqueeze(1)
        sims = torch.where(same, neg, sims)  # impostors only
        top = sims.topk(cohort_n, dim=1).values
        mu[st:e] = top.mean(1)
        sd[st:e] = top.std(1).clamp_min(1e-6)
    return mu, sd


def evaluate(model: str, backend: str, k: int = 10, cohort_n: int = 300, seed: int = 0,
             session_disjoint: bool = False):
    """Closed-set misID / rank-1 / 1:N EER for (model, backend).

    With session_disjoint=True the genuine best-match S_same is restricted to the
    query speaker's own segments from a DIFFERENT source file (source_sha256), so a
    query is only usable if its speaker has >=2 distinct source files; the impostor
    side S_diff (all other speakers) is unchanged. Everything else -- pre-processing,
    AS-norm cohort, scoring loop, bootstrap RNG, sampled EER -- is identical, so
    session_disjoint=False reproduces the published Table-2 values exactly."""
    if session_disjoint:
        emb, spk, sha = load_agency_only(model, with_sha=True)
    else:
        emb, spk = load_agency_only(model)
    X = preprocess(emb, backend)
    Xt = torch.from_numpy(X).to(DEVICE)
    n = len(spk)
    _, codes = np.unique(spk, return_inverse=True)
    spk_t = torch.from_numpy(codes).to(DEVICE)
    sha_t = None
    if session_disjoint:
        _, sha_codes = np.unique(sha, return_inverse=True)
        sha_t = torch.from_numpy(sha_codes).to(DEVICE)

    mu = sd = None
    if backend == "asnorm":
        mu, sd = cohort_stats(Xt, spk_t, cohort_n)

    same_best = np.full(n, -9.0, np.float32)
    diff_best = np.full(n, -9.0, np.float32)
    occ = np.zeros(n, np.int64)
    neg = torch.tensor(-9.0, device=DEVICE)
    chunk = 2048
    for st in range(0, n, chunk):
        e = min(st + chunk, n)
        sims = Xt[st:e] @ Xt.T
        if backend == "asnorm":
            sims = 0.5 * ((sims - mu[st:e].unsqueeze(1)) / sd[st:e].unsqueeze(1)
                          + (sims - mu.unsqueeze(0)) / sd.unsqueeze(0))
        rows = torch.arange(st, e, device=DEVICE)
        sims[torch.arange(e - st, device=DEVICE), rows] = neg
        same = spk_t.unsqueeze(0) == spk_t[st:e].unsqueeze(1)
        genuine = same
        if session_disjoint:
            same_sha = sha_t.unsqueeze(0) == sha_t[st:e].unsqueeze(1)
            genuine = same & ~same_sha  # same speaker AND different source file (excludes self)
        same_best[st:e] = torch.where(genuine, sims, neg).max(1).values.cpu().numpy()
        diff_best[st:e] = torch.where(same, neg, sims).max(1).values.cpu().numpy()
        idx = sims.topk(k, 1).indices.reshape(-1).cpu().numpy()
        np.add.at(occ, idx, 1)

    has = same_best > -8.0
    margin = same_best - diff_best
    # Critical threshold scales with score spread; use the score std for asnorm.
    spread = margin[has].std()
    crit_abs = 0.02 if backend != "asnorm" else 0.1 * spread

    def point(idx):
        m = margin[idx]
        return {
            "rank1": float(np.mean(same_best[idx] > diff_best[idx])),
            "misid": float(np.mean(m < 0)),
            "critical": float(np.mean(np.abs(m) < crit_abs)),
        }

    full = point(has)
    full["hubness_skew"] = float(stats.skew(occ.astype(float)))

    # Speaker-level bootstrap CI (resample speakers with replacement).
    rng = np.random.default_rng(seed)
    uspk = np.unique(codes)
    spk_to_idx = {s: np.where((codes == s) & has)[0] for s in uspk}
    boots = {"rank1": [], "misid": [], "critical": []}
    for _ in range(300):
        samp = rng.choice(uspk, len(uspk), replace=True)
        idx = np.concatenate([spk_to_idx[s] for s in samp if len(spk_to_idx[s])])
        m = margin[idx]
        boots["rank1"].append(np.mean(same_best[idx] > diff_best[idx]))
        boots["misid"].append(np.mean(m < 0))
        boots["critical"].append(np.mean(np.abs(m) < crit_abs))
    ci = {kk: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for kk, v in boots.items()}

    # Verification EER from sampled trials (balanced same/diff pairs).
    eer = sampled_eer(same_best, diff_best, margin, has, codes, rng)
    return {"model": model, "backend": backend, "n": int(n), "crit_abs": float(crit_abs),
            "session_disjoint": session_disjoint,
            "n_query_usable": int(has.sum()), "n_spk_usable": int(len(np.unique(codes[has]))),
            **full, "eer": eer, "ci95": ci}


def sampled_eer(same_best, diff_best, margin, has, codes, rng, n_trials=40000):
    """EER from per-segment genuine (same_best) vs impostor (diff_best) scores.

    Treats each query's best same-speaker score as a target trial and its best
    different-speaker score as a non-target trial -- the operating scores a
    1:N identification-as-verification system actually thresholds.
    """
    idx = np.where(has)[0]
    if len(idx) > n_trials:
        idx = rng.choice(idx, n_trials, replace=False)
    tgt = same_best[idx]
    non = diff_best[idx]
    scores = np.concatenate([tgt, non])
    labels = np.concatenate([np.ones_like(tgt), np.zeros_like(non)])
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum(); N = len(labels) - P
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    fnr = 1 - tp / P
    fpr = fp / N
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2)


def _sd_block(r):
    """Extract the session_disjoint.json per-block schema from an evaluate() result."""
    return {"misid": r["misid"], "eer": r["eer"], "rank1": r["rank1"],
            "n_query_usable": r["n_query_usable"], "n_spk_usable": r["n_spk_usable"]}


def run_session_disjoint(models, backends, out_path):
    """Table-2 closed-set misID + 1:N EER recomputed under a source-disjoint genuine
    constraint. For each (model, backend) emit both the unconstrained `same_session`
    block (reproduces Table-2) and the `session_disjoint` block."""
    out = {"_meta": {
        "note": ("Table-2 (calibrated_eval.py) closed-set misID + 1:N EER, recomputed under a "
                 "session-disjoint constraint: a query's same-speaker match must come from a "
                 "DIFFERENT source file (source_sha256). Impostor (different-speaker) scores "
                 "unchanged. Population: agency-only (drop freelance:), n=45994 segments, matching "
                 "Table 2. session_disjoint=False column reproduces the published Table-2 values "
                 "as a sanity check."),
        "session_key": ("source_sha256 (per source file / recording); joined to npz by segment_id. "
                        "The npz `recording_source` key is agency-level (1 per speaker) and cannot "
                        "define within-speaker sessions, so the per-file source_sha256 is used."),
        "misid_def": ("fraction of query segments whose best different-speaker score exceeds its "
                      "best (session-disjoint) same-speaker score (closed-set rank-1 misID)."),
        "table2_same_session_reference": {},
    }, "results": {}}
    t2 = out["_meta"]["table2_same_session_reference"]
    for m in models:
        out["results"][m] = {}
        t2[m] = {}
        for b in backends:
            ss = evaluate(m, b, session_disjoint=False)
            sd = evaluate(m, b, session_disjoint=True)
            out["results"][m][b] = {"same_session": _sd_block(ss), "session_disjoint": _sd_block(sd)}
            t2[m][b] = {"misid": round(ss["misid"] * 100, 1), "eer": round(ss["eer"] * 100, 1)}
            print(f"{m:9s} {b:7s} misID {ss['misid']*100:5.1f}% -> {sd['misid']*100:5.1f}%  "
                  f"EER {ss['eer']*100:5.1f}% -> {sd['eer']*100:5.1f}%  "
                  f"Nq {ss['n_query_usable']} -> {sd['n_query_usable']}  "
                  f"Nspk {ss['n_spk_usable']} -> {sd['n_spk_usable']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--session-disjoint", action="store_true",
                    help="recompute Table-2 misID/EER/rank1 under a source-disjoint genuine "
                         "constraint (S_same from a different source_sha256); writes session_disjoint.json")
    ap.add_argument("--out", default=None, help="override the output JSON path")
    args = ap.parse_args()

    if args.session_disjoint:
        models = args.models or ["ecapa", "animeva", "ens_sv4"]
        backends = args.backends or ["raw", "asnorm"]
        run_session_disjoint(models, backends, Path(args.out) if args.out else SD_OUT)
        return

    models = args.models or ["ecapa", "campp", "redimnet", "ens_sv4"]
    backends = args.backends or ["raw", "center", "asnorm"]
    results = []
    for m in models:
        for b in backends:
            r = evaluate(m, b)
            results.append(r)
            ci = r["ci95"]
            print(f"{m:9s} {b:7s} rank1={r['rank1']*100:5.1f}% EER={r['eer']*100:4.1f}% "
                  f"crit={r['critical']*100:4.1f}% [{ci['critical'][0]*100:.1f}-{ci['critical'][1]*100:.1f}] "
                  f"misid={r['misid']*100:4.1f}% hub={r['hubness_skew']:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

"""BGM-excluded rerun of the core analyses (paper §7 future-work item).

The PANNs BGM audit (output/analysis/bgm_audit_panns.json) flags 18.6% of
agency source files as containing background music (music_prob > 0.5).
This script re-runs, on the CACHED segment embeddings, with all segments cut
from flagged source files removed:

  (A) Table-1 raw cosine geometry: rank-1 / closed-set misID / hubness skew
      (analyze.py protocol, k=10, full embedding set incl. freelance).
  (B) Table-2 raw -> AS-norm on the codec-matched agency subset
      (calibrated_eval.py protocol: rank-1, 1:N EER, misID, hubness).
  (C) §4.4 real-vs-clone separability (channel_control.py "raw" protocol:
      140 clones/method x 7 methods + 4 real segments/target, logistic
      regression on centered+L2 embeddings, GroupKFold by target, balanced
      accuracy) with BGM-flagged REAL segments excluded. Real embeddings come
      from the cache; clones (music-free, unchanged) are embedded on the fly.

File -> segment mapping: audit row path -> samples.jsonl content_sha256 ->
segments.jsonl source_sha256 -> segment_id (validated: 1,325 flagged files ->
13,971 of 56,592 segments).

Existing analysis scripts are NOT modified. Output:
    output/analysis/bgm_excluded_rerun.json

Usage: python -m src.bgm_excluded_rerun [--models ecapa campp redimnet animeva ens_sv4]
       [--sep-models ecapa animeva] [--skip-separability]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

EMB = ROOT / "output/embeddings"
AUDIT = ROOT / "output/analysis/bgm_audit_panns.json"
OUT = ROOT / "output/analysis/bgm_excluded_rerun.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5
K = 10

CLONES = {  # mirror src/channel_control.py / src/clone_geometry.py
    "seedvc": "/path/to/va-data/clones/seedvc",
    "irodori": "/path/to/va-data/clones/irodori",
    "gptsovits_v1": "/path/to/va-data/clones/gptsovits_v1",
    "gptsovits_v2": "/path/to/va-data/clones/gptsovits_v2",
    "gptsovits_v2ProPlus": "/path/to/va-data/clones/gptsovits",
    "gptsovits_v3": "/path/to/va-data/clones/gptsovits_v3",
    "gptsovits_v4": "/path/to/va-data/clones/gptsovits_v4",
}
MAX_PER_METHOD = 140
REAL_SEGS = 4


# ---------------------------------------------------------------- mapping
def flagged_segment_ids(thresh: float = THRESH) -> set[str]:
    audit = json.load(open(AUDIT))
    flagged_paths = {r["path"] for r in audit["agency_files"] if r["music_prob"] > thresh}
    sha_of = {}
    for l in open(ROOT / "data/agency_voices/samples.jsonl"):
        r = json.loads(l)
        sha_of[str((ROOT / r["local_path"]).resolve())] = r["content_sha256"]
    flagged_shas = {sha_of[p] for p in flagged_paths if p in sha_of}
    seg_ids = set()
    for l in open(ROOT / "data/processed/segments.jsonl"):
        if '"segment_id"' not in l:
            continue
        r = json.loads(l)
        if r.get("source_sha256") in flagged_shas:
            seg_ids.add(r["segment_id"])
    return seg_ids


# ---------------------------------------------------------------- shared metric core
def ln(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def geometry(emb: np.ndarray, spk: np.ndarray, backend: str = "raw",
             cohort_n: int = 300, k: int = K):
    """rank-1 / misID / hubness skew (+1:N EER) under raw cosine or AS-norm.

    Mirrors src/analyze.py (raw) and src/calibrated_eval.py (asnorm)."""
    x = emb.copy()
    if backend == "asnorm":
        x = x - x.mean(0, keepdims=True)
    Xt = torch.from_numpy(ln(x).astype(np.float32)).to(DEVICE)
    n = len(spk)
    _, codes = np.unique(spk, return_inverse=True)
    spk_t = torch.from_numpy(codes).to(DEVICE)
    neg = torch.tensor(-9.0, device=DEVICE)

    mu = sd = None
    if backend == "asnorm":
        mu = torch.empty(n, device=DEVICE)
        sd = torch.empty(n, device=DEVICE)
        for st in range(0, n, 2048):
            e = min(st + 2048, n)
            sims = Xt[st:e] @ Xt.T
            same = spk_t.unsqueeze(0) == spk_t[st:e].unsqueeze(1)
            sims = torch.where(same, neg, sims)
            top = sims.topk(min(cohort_n, n - 1), dim=1).values
            mu[st:e] = top.mean(1)
            sd[st:e] = top.std(1).clamp_min(1e-6)

    same_best = np.full(n, -99.0, np.float32)
    diff_best = np.full(n, -99.0, np.float32)
    occ = np.zeros(n, np.int64)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        sims = Xt[st:e] @ Xt.T
        if backend == "asnorm":
            sims = 0.5 * ((sims - mu[st:e].unsqueeze(1)) / sd[st:e].unsqueeze(1)
                          + (sims - mu.unsqueeze(0)) / sd.unsqueeze(0))
        rows = torch.arange(st, e, device=DEVICE)
        sims[torch.arange(e - st, device=DEVICE), rows] = neg * 999  # mask self
        same = spk_t.unsqueeze(0) == spk_t[st:e].unsqueeze(1)
        same_best[st:e] = torch.where(same, sims, neg * 999).max(1).values.cpu().numpy()
        diff_best[st:e] = torch.where(same, neg * 999, sims).max(1).values.cpu().numpy()
        idx = sims.topk(k, 1).indices.reshape(-1).cpu().numpy()
        np.add.at(occ, idx, 1)

    has = same_best > -999.0
    margin = same_best - diff_best
    # 1:N EER over per-query (S_same, S_diff) pairs -- calibrated_eval.sampled_eer
    tgt, non = same_best[has], diff_best[has]
    s = np.concatenate([tgt, non])
    y = np.concatenate([np.ones_like(tgt), np.zeros_like(non)])
    o = np.argsort(-s); y = y[o]
    P = y.sum(); N = len(y) - P
    fnr = 1 - np.cumsum(y) / P
    fpr = np.cumsum(1 - y) / N
    i = int(np.argmin(np.abs(fnr - fpr)))
    return {
        "n_segments": int(n),
        "n_speakers": int(len(np.unique(spk))),
        "n_queries": int(has.sum()),
        "rank1": float(np.mean(same_best[has] > diff_best[has])),
        "misid": float(np.mean(margin[has] < 0)),
        "eer_1n": float((fnr[i] + fpr[i]) / 2),
        "hubness_skew": float(stats.skew(occ.astype(float))),
    }


def load_model(model: str):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    return (d["emb"].astype(np.float32), np.asarray(d["speaker_id"]),
            np.asarray(d["segment_id"]),
            np.asarray(d["recording_source"]) if "recording_source" in d
            else np.array(["?"] * len(d["emb"])))


def coverage(spk: np.ndarray, keep: np.ndarray):
    """Speaker survival diagnostics for a keep-mask."""
    full = defaultdict(int); kept = defaultdict(int)
    for s, k_ in zip(spk, keep):
        full[s] += 1
        if k_:
            kept[s] += 1
    return {
        "speakers_full": len(full),
        "speakers_kept_ge1": sum(1 for s in full if kept[s] >= 1),
        "speakers_kept_ge2": sum(1 for s in full if kept[s] >= 2),  # rank-1 eligible
        "speakers_kept_ge8": sum(1 for s in full if kept[s] >= 8),  # centroid eligible
        "speakers_full_ge2": sum(1 for s in full if full[s] >= 2),
        "speakers_full_ge8": sum(1 for s in full if full[s] >= 8),
    }


# ---------------------------------------------------------------- (C) separability
def separability(X, y_real, groups):
    """channel_control.separability, verbatim protocol."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import recall_score
    X = X - X.mean(0, keepdims=True)
    X = ln(X)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    bal = cross_val_score(clf, X, y_real, groups=groups, cv=gkf,
                          scoring="balanced_accuracy").mean()
    auc = cross_val_score(clf, X, y_real, groups=groups, cv=gkf,
                          scoring="roc_auc").mean()
    pred = cross_val_predict(clf, X, y_real, groups=groups, cv=gkf)
    return {"balanced_acc": float(bal), "roc_auc": float(auc),
            "recall_real": float(recall_score(y_real, pred, pos_label=1)),
            "recall_synth": float(recall_score(y_real, pred, pos_label=0))}


def run_separability(model: str, flagged: set[str]):
    import soundfile as sf
    from src.clone_probe import va_actor_segments
    from src.embeddings import build_extractor

    actors = va_actor_segments()
    want = set(actors.keys())
    ext = build_extractor(model, DEVICE)

    # clones (music-free; unchanged between conditions)
    clone_rows = []  # (emb, target)
    for method, d in CLONES.items():
        if not Path(d).exists():
            continue
        kept = []
        for f in sorted(Path(d).glob("clone_*__src*.wav")):
            m = re.match(r"clone_(.+)__src\d+", f.stem)
            if m and m.group(1) in want:
                kept.append((f, m.group(1)))
        if len(kept) > MAX_PER_METHOD:
            kept = kept[:: max(1, len(kept) // MAX_PER_METHOD)][:MAX_PER_METHOD]
        for f, t in kept:
            w, sr = sf.read(str(f), dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            assert sr == 16000, f"{f}: expected 16 kHz, got {sr}"
            v = np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)
            clone_rows.append((v, t))
        print(f"  {model}: embedded {method}", flush=True)

    # real side from the CACHED embeddings (identical extractor / files)
    emb, _, seg_ids, _ = load_model(model)
    emb_of = {s: i for i, s in enumerate(seg_ids)}
    targets = {t for _, t in clone_rows}
    real_rows = []  # (emb, target, segment_id)
    for t in targets:
        for p in actors[t][8:8 + REAL_SEGS]:
            sid = Path(p).stem
            if sid in emb_of:
                real_rows.append((emb[emb_of[sid]], t, sid))

    def assemble(exclude_flagged: bool):
        rows_r = [r for r in real_rows if not (exclude_flagged and r[2] in flagged)]
        X = np.stack([v for v, _ in clone_rows] + [v for v, _, _ in rows_r])
        y = np.concatenate([np.zeros(len(clone_rows)), np.ones(len(rows_r))]).astype(int)
        g = np.array([t for _, t in clone_rows] + [t for _, t, _ in rows_r])
        res = separability(X.astype(np.float32), y, g)
        res.update(n_clone=len(clone_rows), n_real=len(rows_r),
                   n_targets_with_real=len({t for _, t, _ in rows_r}))
        return res

    full = assemble(False)
    excl = assemble(True)
    return {"full": full, "bgm_excluded": excl}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["ecapa", "campp", "redimnet", "animeva", "ens_sv4"])
    ap.add_argument("--sep-models", nargs="+", default=["ecapa", "animeva"])
    ap.add_argument("--skip-separability", action="store_true")
    args = ap.parse_args()

    flagged = flagged_segment_ids()
    print(f"flagged segments: {len(flagged)}", flush=True)

    out = {"threshold": THRESH, "n_flagged_segments": len(flagged),
           "table1_raw_geometry": {}, "table2_raw_asnorm_codec_matched": {},
           "separability_real_vs_clone": {}}

    for m in args.models:
        emb, spk, seg_ids, src = load_model(m)
        keep_bgm = ~np.isin(seg_ids, list(flagged))

        # (A) Table 1: full embedding set (incl. freelance), raw cosine
        cov = coverage(spk, keep_bgm)
        out["table1_raw_geometry"][m] = {
            "coverage": cov,
            "full": geometry(emb, spk, "raw"),
            "bgm_excluded": geometry(emb[keep_bgm], spk[keep_bgm], "raw"),
        }

        # (B) Table 2: codec-matched agency subset (no freelance), raw -> AS-norm
        agency = np.array([not str(s).startswith("freelance:") for s in src])
        ka = agency & keep_bgm
        out["table2_raw_asnorm_codec_matched"][m] = {
            "coverage": coverage(spk[agency], keep_bgm[agency]),
            "full": {b: geometry(emb[agency], spk[agency], b) for b in ("raw", "asnorm")},
            "bgm_excluded": {b: geometry(emb[ka], spk[ka], b) for b in ("raw", "asnorm")},
        }
        t1 = out["table1_raw_geometry"][m]
        print(f"{m}: T1 rank1 {t1['full']['rank1']*100:.1f} -> {t1['bgm_excluded']['rank1']*100:.1f} "
              f"| hub {t1['full']['hubness_skew']:.2f} -> {t1['bgm_excluded']['hubness_skew']:.2f}",
              flush=True)

    if not args.skip_separability:
        for m in args.sep_models:
            print(f"=== separability {m} ===", flush=True)
            out["separability_real_vs_clone"][m] = run_separability(m, flagged)
            s = out["separability_real_vs_clone"][m]
            print(f"{m}: bal-acc {s['full']['balanced_acc']*100:.1f} -> "
                  f"{s['bgm_excluded']['balanced_acc']*100:.1f} "
                  f"(real n {s['full']['n_real']} -> {s['bgm_excluded']['n_real']})",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

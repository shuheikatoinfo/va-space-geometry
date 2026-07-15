"""Shared centered-cosine gallery for the clone probe and the open-set
false-match experiments (amateur / fictional).

Builds, from the CACHED segment embeddings (output/embeddings/{model}.npz),
the same gallery score_clones.py historically built by re-embedding audio:
per-actor centroids from the first `n_enroll` agency segments (L2-normed
mean), mean-centered across centroids and re-normalized, plus a real-vs-real
EER threshold computed in that SAME centered space. Every consumer of the
threshold must score queries in this coordinate system; applying a
centered-space threshold to uncentered cosines (the bug this module fixes)
inflates false-match rates by a coordinate-system artifact.

Two thresholds are exposed:
  thr        : genuine probes = segments [n_enroll : n_enroll+n_gen] in file
               order (they may share the source file with enrollment ->
               same-session-optimistic; the historical protocol)
  thr_xfile  : genuine probes drawn only from source files (source_sha256)
               NOT used in enrollment (cross-file; deployment-realistic)

Query vectors are centered exactly as score_clones centers clone embeddings:
v_c = (v_raw - mean) / ||v_raw - mean||  (no pre-L2-normalization of v_raw,
matching the published protocol).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB = Path("output/embeddings")
SEGMENTS = Path("data/processed/segments.jsonl")


def eer_threshold(gen, imp):
    s = np.concatenate([gen, imp])
    lb = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s)
    lb = lb[o]; s = s[o]
    P, N = lb.sum(), len(lb) - lb.sum()
    tp = np.cumsum(lb); fp = np.cumsum(1 - lb)
    fnr = 1 - tp / P; fpr = fp / N
    i = np.argmin(np.abs(fnr - fpr))
    return float(s[i])


def agency_actor_segments(min_segs=8):
    """Per-speaker ordered (segment_id, source_sha256, segment_path) lists,
    agency-studio audio only, speakers with >= min_segs segments.
    segments.jsonl contains a handful of duplicate segment_id lines (same audio,
    repeated metadata row); dedupe so no segment can appear both inside a
    centroid and as its own genuine probe."""
    by = defaultdict(list)
    seen = set()
    for l in open(SEGMENTS, encoding="utf-8"):
        if '"segment_id"' not in l:
            continue
        r = json.loads(l)
        if r["segment_id"] in seen:
            continue
        seen.add(r["segment_id"])
        if not r["recording_source"].startswith("freelance:"):
            by[r["speaker_id"]].append((r["segment_id"], r["source_sha256"], r["segment_path"]))
    return {s: v for s, v in by.items() if len(v) >= min_segs}


def center_query(v_raw, mean):
    v = np.asarray(v_raw, np.float32).reshape(-1) - mean
    return v / max(np.linalg.norm(v), 1e-9)


def centered_gallery(model, n_enroll=8, n_gen=4):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    embn = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    row_of = {sid: i for i, sid in enumerate(d["segment_id"].astype(str))}
    actors = agency_actor_segments(min_segs=n_enroll)
    actor_ids = sorted(actors)
    C, gen_rows, gen_xf_rows = [], {}, {}
    for a in actor_ids:
        segs = actors[a]
        enroll = segs[:n_enroll]
        C.append(embn[[row_of[sid] for sid, _, _ in enroll]].mean(0))
        gen_rows[a] = [row_of[sid] for sid, _, _ in segs[n_enroll:n_enroll + n_gen]]
        efiles = {sha for _, sha, _ in enroll}
        xf = [row_of[sid] for sid, sha, _ in segs[n_enroll:] if sha not in efiles]
        gen_xf_rows[a] = xf[:n_gen]
    C = np.stack(C)
    mean = C.mean(0, keepdims=True)
    Cn = (C - mean) / np.clip(np.linalg.norm(C - mean, axis=1, keepdims=True), 1e-9, None)
    mean = mean.squeeze(0)
    idx_of = {a: i for i, a in enumerate(actor_ids)}

    def scores_from(rows_map):
        gen, imp = [], []
        for a, rows in rows_map.items():
            if not rows:
                continue
            es = emb[rows] - mean
            es = es / np.clip(np.linalg.norm(es, axis=1, keepdims=True), 1e-9, None)
            sims = es @ Cn.T
            gen += list(sims[:, idx_of[a]])
            imp += list(np.delete(sims, idx_of[a], 1).max(1))
        return np.array(gen), np.array(imp)

    gen, imp = scores_from(gen_rows)
    gen_xf, imp_xf = scores_from(gen_xf_rows)
    return {
        "actor_ids": actor_ids, "idx_of": idx_of, "Cn": Cn, "mean": mean,
        "thr": eer_threshold(gen, imp),
        "thr_xfile": eer_threshold(gen_xf, imp_xf),
        "gen_scores": gen, "gen_scores_xfile": gen_xf,
        "n_gen_trials": len(gen), "n_xfile_trials": len(gen_xf),
        "n_xfile_actors": sum(1 for v in gen_xf_rows.values() if v),
        "segments": actors,
    }

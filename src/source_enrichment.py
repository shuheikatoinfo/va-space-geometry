"""Peer-review control: source-speaker (jvs001) enrichment test.

All Seed-VC clones were converted from the SAME six jvs001 (JVS corpus) source
utterances. A reviewer concern: residual SOURCE-speaker traces could drive the
wrongful-attribution results (clones landing on innocent actors merely because
those actors resemble jvs001). Control: embed the six source utterances, build
a jvs001 query vector per encoder in the same centered-cosine gallery space as
score_clones.py, rank all ~1,100 agency-actor centroids by similarity to
jvs001, and test whether the wrongly-attributed actors from
output/analysis/clone_score_seedvc.json are enriched near jvs001
(Mann-Whitney U on ranks; hypergeometric top-10/top-50 overlap).

Usage:
    python -m src.source_enrichment --models animeva ecapa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import mannwhitneyu, hypergeom

from src.centered_gallery import centered_gallery, center_query
from src.embeddings import build_extractor

JVS_SEG = Path("/path/to/va-data/processed/jvs")
SCORES = Path("output/analysis/clone_score_seedvc.json")
OUT = Path("output/analysis/source_enrichment.json")


def pick_sources(n=6):
    """Exactly replicates clones/run_seedvc.py:pick_sources — the first n wavs
    (sorted) in the JVS segment dir with duration >= 2 s."""
    src = []
    for p in sorted(JVS_SEG.glob("*.wav")):
        try:
            info = sf.info(str(p))
            if info.frames / info.samplerate >= 2.0:
                src.append(p)
        except Exception:
            continue
        if len(src) >= n:
            break
    return src


def emb_file(ext, p):
    w, sr = sf.read(str(p), dtype="float32")
    assert sr == 16000, f"{p}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
    if w.ndim > 1:
        w = w.mean(1)
    return np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    args = ap.parse_args()

    sources = pick_sources(6)
    assert len(sources) == 6, f"expected 6 sources, got {len(sources)}"
    assert all("jvs001" in p.name for p in sources), \
        "premise violated: sources are not all jvs001"
    print("source utterances (same six as clones/run_seedvc.py):")
    for p in sources:
        print(f"  {p.name}")

    scores = json.load(open(SCORES, encoding="utf-8"))
    results = {"sources": [str(p) for p in sources]}

    for model in args.models:
        gal = centered_gallery(model)
        actor_ids, idx_of = gal["actor_ids"], gal["idx_of"]
        Cn, mean = gal["Cn"], gal["mean"]
        G = len(actor_ids)

        # jvs001 vector: per-utterance L2-norm then mean (centroid convention),
        # centered+L2 with the gallery mean like every query in score_clones.
        ext = build_extractor(model, "cuda")
        es = np.stack([emb_file(ext, p) for p in sources])
        es = es / np.clip(np.linalg.norm(es, axis=1, keepdims=True), 1e-9, None)
        jvs = center_query(es.mean(0), mean)

        sims = Cn @ jvs
        order = np.argsort(-sims)                       # actor indices, nearest first
        rank_of = np.empty(G, int)
        rank_of[order] = np.arange(1, G + 1)            # rank 1 = nearest to jvs001

        rows = scores[model]["rows"]
        wrong_rows = [r for r in rows if r["false"]]
        wrong_actors = sorted({r["nearest"] for r in wrong_rows})
        wrong_idx = [idx_of[a] for a in wrong_actors if a in idx_of]
        w_ranks = rank_of[wrong_idx].astype(float)
        other_idx = [i for i in range(G) if i not in set(wrong_idx)]
        o_ranks = rank_of[other_idx].astype(float)

        # Mann-Whitney: are wrong-actor ranks smaller (closer to jvs001)?
        mw = mannwhitneyu(w_ranks, o_ranks, alternative="less")

        # weighted view: each wrongful attribution event counted
        ev_ranks = np.array([rank_of[idx_of[r["nearest"]]] for r in wrong_rows
                             if r["nearest"] in idx_of], float)

        def topk(k):
            hits = int((w_ranks <= k).sum())
            expected = k * len(w_ranks) / G
            pv = float(hypergeom.sf(hits - 1, G, k, len(w_ranks)))
            ev_hits = int((ev_ranks <= k).sum())
            wr_ranks = np.array([rank_of[idx_of[r["nearest"]]] for r in wrong_rows
                                 if r["wrongful"] and r["nearest"] in idx_of], float)
            return {"k": k, "hits": hits, "expected": round(expected, 2),
                    "p_hypergeom": pv,
                    "events_in_topk": ev_hits, "events_total": len(ev_ranks),
                    "events_chance_rate": k / G,
                    "wrongful_over_thr_events_in_topk": int((wr_ranks <= k).sum()),
                    "wrongful_over_thr_events_total": len(wr_ranks)}

        res = {
            "n_gallery": G,
            "n_wrong_attribution_rows": len(wrong_rows),
            "n_unique_wrong_actors": len(wrong_actors),
            "wrong_actors": wrong_actors,
            "wrong_actor_ranks": {a: int(rank_of[idx_of[a]]) for a in wrong_actors if a in idx_of},
            "median_rank_wrong": float(np.median(w_ranks)) if len(w_ranks) else None,
            "median_percentile_wrong": float(np.median(w_ranks) / G * 100) if len(w_ranks) else None,
            "median_rank_all": float((G + 1) / 2),
            "median_rank_events_weighted": float(np.median(ev_ranks)) if len(ev_ranks) else None,
            "mannwhitney_U": float(mw.statistic),
            "mannwhitney_p_one_sided_less": float(mw.pvalue),
            "top10": topk(10),
            "top50": topk(50),
            "sim_jvs001_to_nearest_gallery": float(sims[order[0]]),
            "nearest_gallery_actor_to_jvs001": actor_ids[order[0]],
        }
        results[model] = res

        print(f"\n## {model} (gallery={G}) ##")
        print(f"  wrong-attribution clones: {len(wrong_rows)}  "
              f"unique wrongly-attributed actors: {len(wrong_actors)}")
        print(f"  median rank (cosine to jvs001) of wrong actors: "
              f"{res['median_rank_wrong']:.0f} / {G} "
              f"({res['median_percentile_wrong']:.1f}th pct; chance = 50th)")
        print(f"  event-weighted median rank: {res['median_rank_events_weighted']:.0f}")
        print(f"  Mann-Whitney U (wrong closer to jvs001?): p = "
              f"{res['mannwhitney_p_one_sided_less']:.4g}")
        for t in (res["top10"], res["top50"]):
            print(f"  overlap with jvs001 top-{t['k']}: {t['hits']} "
                  f"(expected {t['expected']}, hypergeom p = {t['p_hypergeom']:.4g})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

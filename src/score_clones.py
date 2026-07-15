"""Score pre-generated voice clones against the FULL real-actor gallery.

Unlike a closed gallery of only the cloned actors, here the gallery is every
agency actor with >= 8 real segments (~1,100), scored in the shared centered
cosine space (src/centered_gallery.py; gallery + thresholds come from the
cached segment embeddings, so real audio is never re-embedded).

Metrics per clone of target T (nearest = argmax over all gallery actors):
  det          sim_to_T >= thr                      (missed-detection view)
  false        nearest != T                          (nearest-neighbour label
               mismatch, UNCONDITIONAL -- includes sub-threshold clones a real
               detector would reject; kept for continuity)
  wrongful     best_other >= thr AND nearest != T    (deployment-relevant
               wrongful accusation: an INNOCENT actor is both top-ranked and
               over the operating threshold)
  collat       #innocent actors over thr             (collateral accusations)

Every rate is reported at two operating points: thr (genuine trials may share
the enrollment source file; same-session-optimistic, the historical protocol)
and thr_xfile (cross-file genuine trials; deployment-realistic, lower ->
detection easier, false matches more frequent).

Usage:
    python -m src.score_clones --clone-dir /path/to/va-data/clones/seedvc --models animeva ecapa
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from src.centered_gallery import centered_gallery, center_query
from src.embeddings import build_extractor

RATE_KEYS = ["det", "det_xf", "corr", "false", "wrongful", "wrongful_xf",
             "thin", "collat", "collat_xf"]


def emb_file(ext, p):
    w, sr = sf.read(p, dtype="float32")
    assert sr == 16000, f"{p}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
    if w.ndim > 1:
        w = w.mean(1)
    return np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or Path(args.clone_dir).name

    clones = sorted(Path(args.clone_dir).glob("clone_*__src*.wav"))

    results = {}
    for model in args.models:
        gal = centered_gallery(model)
        actor_ids, idx_of = gal["actor_ids"], gal["idx_of"]
        Cn, mean = gal["Cn"], gal["mean"]
        thr, thr_xf = gal["thr"], gal["thr_xfile"]
        clone_targets = []
        for c in clones:
            m = re.match(r"clone_(.+)__src\d+", c.stem)
            if m and m.group(1) in idx_of:
                clone_targets.append((m.group(1), str(c)))
        print(f"gallery: {len(actor_ids)} actors | clones: {len(clone_targets)} | "
              f"thr {thr:.4f} (same-file genuine) thr_xfile {thr_xf:.4f} "
              f"({gal['n_xfile_actors']} actors with cross-file genuine trials)")

        ext = build_extractor(model, "cuda")
        rows = []
        for T, path in clone_targets:
            v = center_query(emb_file(ext, path), mean)
            sims = Cn @ v
            simT = float(sims[idx_of[T]])
            others = np.delete(sims, idx_of[T])
            best_other = float(others.max())
            nearest = actor_ids[int(np.argmax(sims))]
            rows.append({
                "target": T, "nearest": nearest,
                "sim_to_T": simT, "best_other": best_other,
                "det": int(simT >= thr), "det_xf": int(simT >= thr_xf),
                "corr": int(nearest == T), "false": int(nearest != T),
                "wrongful": int(best_other >= thr and nearest != T),
                "wrongful_xf": int(best_other >= thr_xf and nearest != T),
                "thin": int(abs(simT - best_other) < 0.05),
                "collat": int((others >= thr).sum()),
                "collat_xf": int((others >= thr_xf).sum()),
            })
        n = len(rows)

        def rate(rs, key):
            return float(np.mean([r[key] for r in rs])) if rs else float("nan")

        # speaker-level bootstrap CI (resample target speakers with replacement)
        by_t = defaultdict(list)
        for r in rows:
            by_t[r["target"]].append(r)
        uts = list(by_t)
        rng = np.random.default_rng(0)
        boot = defaultdict(list)
        for _ in range(1000):
            samp = rng.choice(uts, len(uts), replace=True)
            rs = [r for t in samp for r in by_t[t]]
            for key in RATE_KEYS:
                boot[key].append(rate(rs, key))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

        results[model] = {
            "n": n, "n_target_speakers": len(uts),
            "n_gallery": len(actor_ids),
            "threshold": thr, "threshold_xfile": thr_xf,
            "n_xfile_actors": gal["n_xfile_actors"],
            "detectable": rate(rows, "det"), "detectable_xfile": rate(rows, "det_xf"),
            "correct_attr": rate(rows, "corr"),
            "false_attr": rate(rows, "false"),
            "wrongful_accusation": rate(rows, "wrongful"),
            "wrongful_accusation_xfile": rate(rows, "wrongful_xf"),
            "thin_margin": rate(rows, "thin"),
            "mean_collateral": rate(rows, "collat"),
            "mean_collateral_xfile": rate(rows, "collat_xf"),
            "ci95": ci,
            "rows": rows,
        }
        r = results[model]
        print(f"## {label} / {model} (gallery={len(actor_ids)}, n={n}, {len(uts)} targets) ##")
        print(f"  detectable@EER:        {r['detectable']*100:5.1f}%  CI[{ci['det'][0]*100:.0f}-{ci['det'][1]*100:.0f}]"
              f"   (cross-file thr: {r['detectable_xfile']*100:.1f}%)")
        print(f"  nearest!=T (uncond.):  {r['false_attr']*100:5.1f}%  CI[{ci['false'][0]*100:.0f}-{ci['false'][1]*100:.0f}]")
        print(f"  WRONGFUL accusation:   {r['wrongful_accusation']*100:5.1f}%  CI[{ci['wrongful'][0]*100:.0f}-{ci['wrongful'][1]*100:.0f}]"
              f"   (cross-file thr: {r['wrongful_accusation_xfile']*100:.1f}%)")
        print(f"  thin-margin clones:    {r['thin_margin']*100:5.1f}%")
        print(f"  mean collateral:       {r['mean_collateral']:.2f}  (cross-file thr: {r['mean_collateral_xfile']:.2f})")
    out = Path(f"output/analysis/clone_score_{label}.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

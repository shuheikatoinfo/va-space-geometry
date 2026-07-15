"""Sweep the operating threshold tau over real-vs-real impostor FAR quantiles
(plus the EER point) for each (synthesizer, encoder), reporting detection recall,
wrongful accusation, and mean collateral at each operating point.

Reproduces output/analysis/tau_sweep.json (Table tb:tau), which was previously
produced by an ad-hoc script with no committed source. It reuses the shared
centered-cosine gallery (src/centered_gallery.py -- the same gallery and
real-vs-real EER threshold as src/score_clones.py) and the per-clone scoring of
src/score_clones.py. Real audio is never re-embedded (cached gallery vectors);
only the clone WAVs are embedded on the fly, so seed-pinned clone regeneration is
reflected here by simply re-running this module.

The FAR-quantile thresholds and the EER threshold depend only on the (unchanged)
real-audio impostor distribution, so tau values are stable across a clone rerun;
detection_recall / wrongful_accusation / mean_collateral move with the clones.

Usage:
    python -m src.tau_sweep --synths seedvc irodori120 --models animeva ecapa
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.centered_gallery import centered_gallery, center_query
from src.embeddings import build_extractor
from src.score_clones import emb_file

CLONES = Path("/path/to/va-data/clones")
# tau_sweep synthesizer label -> clone directory name
SYNTH_DIR = {"seedvc": "seedvc", "irodori120": "irodori"}
FAR_TARGETS = [0.01, 0.02, 0.05, 0.10]
PROTOCOL = ("same-file EER centered-cosine gallery (score_clones.py); "
            "tau swept over real-vs-real impostor FAR quantiles")


def score_dir(gal, clone_dir, ext):
    """Per-clone: sim to true target, best impostor sim, NN-mismatch flag, and
    the full vector of sims to non-target actors (for collateral counts)."""
    actor_ids, idx_of = gal["actor_ids"], gal["idx_of"]
    Cn, mean = gal["Cn"], gal["mean"]
    simT, best_other, mism, others_rows, targets = [], [], [], [], []
    for c in sorted(Path(clone_dir).glob("clone_*__src*.wav")):
        m = re.match(r"clone_(.+)__src\d+", c.stem)
        if not (m and m.group(1) in idx_of):
            continue
        T = m.group(1)
        v = center_query(emb_file(ext, str(c)), mean)
        sims = Cn @ v
        others = np.delete(sims, idx_of[T])
        simT.append(float(sims[idx_of[T]]))
        best_other.append(float(others.max()))
        mism.append(actor_ids[int(np.argmax(sims))] != T)
        others_rows.append(others)
        targets.append(T)
    return (np.array(simT), np.array(best_other), np.array(mism, bool),
            np.stack(others_rows), targets)


def op_point(label, far_target, tau, imp, simT, best_other, mism, others_mat):
    tau = float(tau)
    return {
        "label": label,
        "far_target": far_target,
        "tau": tau,
        "far_actual": float((imp >= tau).mean()),
        "detection_recall": float((simT >= tau).mean()),
        "wrongful_accusation": float(((best_other >= tau) & mism).mean()),
        "false_attr": float(mism.mean()),
        "mean_collateral": float((others_mat >= tau).sum(1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synths", nargs="+", default=["seedvc", "irodori120"])
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--out", default="output/analysis/tau_sweep.json")
    args = ap.parse_args()

    out = {"protocol": PROTOCOL, "synthesizers": {}}
    for syn in args.synths:
        cdir = CLONES / SYNTH_DIR[syn]
        out["synthesizers"][syn] = {}
        for model in args.models:
            gal = centered_gallery(model)
            imp = np.asarray(gal["imp_scores"], np.float64)
            thr = float(gal["thr"])
            ext = build_extractor(model, "cuda")
            simT, best_other, mism, others_mat, targets = score_dir(gal, str(cdir), ext)

            ops = [op_point(f"FAR~{int(f * 100)}%", f, float(np.quantile(imp, 1.0 - f)),
                            imp, simT, best_other, mism, others_mat)
                   for f in FAR_TARGETS]
            ops.append(op_point("EER", None, thr, imp,
                                simT, best_other, mism, others_mat))

            out["synthesizers"][syn][model] = {
                "n_clones": int(len(simT)),
                "n_target_speakers": int(len(set(targets))),
                "n_gallery": int(len(gal["actor_ids"])),
                "eer_threshold": thr,
                "eer_far": float((imp >= thr).mean()),
                "false_attr_rate": float(mism.mean()),
                "operating_points": ops,
            }
            e = ops[-1]
            print(f"[{syn}/{model}] n={len(simT)} gallery={len(gal['actor_ids'])} "
                  f"EER thr={thr:.4f}  recall@EER={e['detection_recall'] * 100:5.1f}%  "
                  f"wrongful@EER={e['wrongful_accusation'] * 100:5.2f}%  "
                  f"far@EER={e['far_actual'] * 100:.1f}%", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

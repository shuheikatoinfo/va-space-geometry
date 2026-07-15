"""Non-mated clone probe: the direct measurement of the "scenario 1" false
alarm from §1 — a clone of a person who is NOT an enrolled voice actor happens
to match an enrolled actor above the operating threshold.

Targets are the corpus's FREELANCE speakers (recording_source starts with
"freelance:"). The clone-probe gallery (src/centered_gallery.py) is built from
agency-studio audio only, so freelance-only speakers are genuinely NOT
enrolled: every above-threshold top-1 hit against the gallery is a false alarm
(a wrongful accusation of an enrolled actor triggered by a non-enrolled
person's clone). Speakers that also have >= 8 agency segments (and hence a
gallery centroid) are excluded when the manifest is built.

Scored exactly like src/score_clones.py: agency per-actor centroids (mean of
first 8 agency segments), gallery-mean centering + renorm, cosine, EER
operating thresholds thr (same-file genuine; historical protocol) and
thr_xfile (cross-file genuine; deployment-realistic).

Metrics per probe (no enrolled target exists, so there is no "correct" answer):
  fa / fa_xf         top-1 gallery score >= thr / thr_xfile  (false alarm)
  collat / collat_xf #gallery actors >= threshold            (collateral)
  top1_margin        top-1 score minus thr (distribution reported)

Probe sets:
  - clones per synthesizer (--clone-dirs, wavs named clone_<spk>__srcNN.wav)
  - the targets' REAL freelance segments (cached embeddings; the non-mated
    real-audio false-alarm baseline that shares the freelance channel)

Caveat (report with the numbers): freelance audio comes from a different
channel (YouTube / personal-site codecs) than the agency gallery, which can
DEFLATE similarity to agency centroids and make these false-alarm rates
optimistic. The real-segment baseline shares that channel; the clones are
synthesizer-channel, so the clone numbers are less affected.

Usage:
    python -m src.nonmated_probe \
        --clone-dirs /path/to/va-data/clones/nonmated_irodori \
                     /path/to/va-data/clones/nonmated_seedvc \
        --models animeva ecapa
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.centered_gallery import centered_gallery, center_query
from src.embeddings import build_extractor

MANIFEST = Path("/path/to/va-data/nonmated_targets.json")
OUT = Path("output/analysis/nonmated_probe.json")
EMB = Path("output/embeddings")
RATE_KEYS = ["fa", "fa_xf", "collat", "collat_xf"]


def emb_file(ext, p):
    w, sr = sf.read(p, dtype="float32")
    assert sr == 16000, f"{p}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
    if w.ndim > 1:
        w = w.mean(1)
    return np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)


def score_rows(vecs_by_target, gal):
    """vecs_by_target: {speaker_id: [raw embedding vectors]} -> per-probe rows."""
    Cn, mean = gal["Cn"], gal["mean"]
    thr, thr_xf = gal["thr"], gal["thr_xfile"]
    actor_ids = gal["actor_ids"]
    rows = []
    for T, vecs in vecs_by_target.items():
        for v_raw in vecs:
            v = center_query(v_raw, mean)
            sims = Cn @ v
            top1 = float(sims.max())
            rows.append({
                "target": T,
                "top1": top1,
                "top1_actor": actor_ids[int(np.argmax(sims))],
                "fa": int(top1 >= thr), "fa_xf": int(top1 >= thr_xf),
                "collat": int((sims >= thr).sum()),
                "collat_xf": int((sims >= thr_xf).sum()),
                "margin": top1 - thr, "margin_xf": top1 - thr_xf,
            })
    return rows


def summarize(rows, gal, n_boot=1000):
    thr, thr_xf = gal["thr"], gal["thr_xfile"]
    n = len(rows)
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["target"]].append(r)
    uts = sorted(by_t)

    def rate(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    rng = np.random.default_rng(0)
    boot = defaultdict(list)
    for _ in range(n_boot):
        samp = rng.choice(uts, len(uts), replace=True)
        rs = [r for t in samp for r in by_t[t]]
        for key in RATE_KEYS:
            boot[key].append(rate(rs, key))
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in boot.items()}

    margins = np.array([r["margin"] for r in rows])
    margins_xf = np.array([r["margin_xf"] for r in rows])
    top1 = np.array([r["top1"] for r in rows])
    # per-target speaker: any probe of this speaker falsely alarms
    spk_fa = float(np.mean([any(r["fa"] for r in by_t[t]) for t in uts]))
    spk_fa_xf = float(np.mean([any(r["fa_xf"] for r in by_t[t]) for t in uts]))
    return {
        "n_probes": n, "n_target_speakers": len(uts),
        "false_alarm_rate": rate(rows, "fa"),
        "false_alarm_rate_xfile": rate(rows, "fa_xf"),
        "any_probe_false_alarm_per_speaker": spk_fa,
        "any_probe_false_alarm_per_speaker_xfile": spk_fa_xf,
        "mean_collateral": rate(rows, "collat"),
        "mean_collateral_xfile": rate(rows, "collat_xf"),
        "top1_score": {"mean": float(top1.mean()),
                       "p50": float(np.percentile(top1, 50)),
                       "p90": float(np.percentile(top1, 90)),
                       "p99": float(np.percentile(top1, 99)),
                       "max": float(top1.max())},
        "top1_margin_vs_thr": {"mean": float(margins.mean()),
                               "p50": float(np.percentile(margins, 50)),
                               "p90": float(np.percentile(margins, 90)),
                               "p99": float(np.percentile(margins, 99)),
                               "max": float(margins.max())},
        "top1_margin_vs_thr_xfile": {"mean": float(margins_xf.mean()),
                                     "p90": float(np.percentile(margins_xf, 90)),
                                     "max": float(margins_xf.max())},
        "ci95": ci,
        "threshold": thr, "threshold_xfile": thr_xf,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dirs", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    targets = json.load(open(MANIFEST, encoding="utf-8"))
    tids = {str(t["target"]) for t in targets}

    results = {"manifest": str(MANIFEST), "n_manifest_targets": len(targets)}
    for model in args.models:
        gal = centered_gallery(model)
        overlap = tids & set(gal["idx_of"])
        assert not overlap, f"non-mated targets present in gallery: {overlap}"

        # Real-segment baseline from the cached segment embeddings.
        d = np.load(EMB / f"{model}.npz", allow_pickle=True)
        row_of = {sid: i for i, sid in enumerate(d["segment_id"].astype(str))}
        emb = d["emb"].astype(np.float32)
        real = {}
        for t in targets:
            rows_ = [row_of[s] for s in t["real_probe_sids"] if s in row_of]
            if rows_:
                real[str(t["target"])] = [emb[i] for i in rows_]

        mres = {
            "n_gallery": len(gal["actor_ids"]),
            "threshold": gal["thr"], "threshold_xfile": gal["thr_xfile"],
            "n_xfile_actors": gal["n_xfile_actors"],
            "real_baseline": summarize(score_rows(real, gal), gal),
        }

        ext = build_extractor(model, "cuda")
        for cd in args.clone_dirs:
            cd = Path(cd)
            label = cd.name.replace("nonmated_", "")
            by_t = defaultdict(list)
            for p in sorted(cd.glob("clone_*__src*.wav")):
                m = re.match(r"clone_(.+)__src\d+", p.stem)
                if m and m.group(1) in tids:
                    by_t[m.group(1)].append(emb_file(ext, str(p)))
            mres[label] = summarize(score_rows(by_t, gal), gal)

        for label, s in mres.items():
            if not isinstance(s, dict) or "false_alarm_rate" not in s:
                continue
            ci = s["ci95"]
            print(f"## {model} / {label} (n={s['n_probes']}, {s['n_target_speakers']} speakers) ##")
            print(f"  false alarm (top1>=thr):     {s['false_alarm_rate']*100:5.1f}%  "
                  f"CI[{ci['fa'][0]*100:.1f}-{ci['fa'][1]*100:.1f}]  "
                  f"(xfile thr: {s['false_alarm_rate_xfile']*100:.1f}% "
                  f"CI[{ci['fa_xf'][0]*100:.1f}-{ci['fa_xf'][1]*100:.1f}])")
            print(f"  mean collateral >= thr:      {s['mean_collateral']:.3f}  "
                  f"(xfile: {s['mean_collateral_xfile']:.3f})")
            print(f"  top1 margin vs thr mean/p90/max: {s['top1_margin_vs_thr']['mean']:.3f} / "
                  f"{s['top1_margin_vs_thr']['p90']:.3f} / {s['top1_margin_vs_thr']['max']:.3f}")
        results[model] = mres

    results["caveats"] = [
        "Freelance target audio is a different channel (YouTube/personal-site codecs) than the "
        "agency-studio gallery; channel mismatch can deflate similarity to agency centroids, so "
        "the real-segment baseline false-alarm rates may be optimistic (too low).",
        "The clones are synthesizer-channel audio (Irodori-TTS / Seed-VC output), not "
        "freelance-channel, so the clone false-alarm rates are less affected by the freelance "
        "channel confound than the real baseline; the clone-vs-real gap is therefore a lower "
        "bound on the effect of the cloning process under a shared-channel comparison.",
        "Thresholds are the real-vs-real EER operating points of the mated clone probe "
        "(centered-cosine space, agency gallery); thr uses same-file genuine trials "
        "(historical protocol), thr_xfile uses cross-file genuine trials (deployment-realistic).",
    ]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp, "w"), ensure_ascii=False, indent=2)
    print(f"-> {outp}")


if __name__ == "__main__":
    main()

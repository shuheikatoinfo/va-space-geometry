"""Recompute DNN-MOS with UTMOSv2 (2024) as a naturalness sanity check.

Companion to the original UTMOS22-strong pass recorded in
``output/analysis/mos_wer.json``. This script re-scores the same speaker
groups (real demo-reel audio of the 120 clone targets, plus their Seed-VC /
Irodori-TTS / GPT-SoVITS clones) with the newer UTMOSv2 predictor so the paper
can report the latest model. Same sampling design as the original pass
(``sample_per_group`` files per group, fixed seed) so the numbers are
comparable in structure; only the MOS model changes.

Clone audio lives outside the repo (not redistributed); point --clones-root at
it. Real audio is read from data/processed/<target>/*.wav.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import utmosv2


def group_files(clones_root, repo_root, targets):
    real_pool = []
    for t in targets:
        real_pool += glob.glob(os.path.join(repo_root, "data", "processed", t, "*.wav"))
    return {
        "real": real_pool,
        "seedvc": glob.glob(os.path.join(clones_root, "seedvc", "*.wav")),
        "irodori": glob.glob(os.path.join(clones_root, "irodori", "*.wav")),
        "gptsovits_v1": glob.glob(os.path.join(clones_root, "gptsovits_v1", "*.wav")),
        "gptsovits_v2": glob.glob(os.path.join(clones_root, "gptsovits_v2", "*.wav")),
        "gptsovits_v3": glob.glob(os.path.join(clones_root, "gptsovits_v3", "*.wav")),
        "gptsovits_v4": glob.glob(os.path.join(clones_root, "gptsovits_v4", "*.wav")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clones-root", default="/path/to/va-data/clones")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--sample-per-group", type=int, default=144)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--folds", default="0,1,2,3,4",
                    help="UTMOSv2 folds to ensemble (official pretrained = 5-fold)")
    ap.add_argument("--out", default="output/analysis/mos_utmosv2.json")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    targets = [t["target"] for t in json.load(
        open(os.path.join(args.clones_root, os.pardir, "clone_targets.json")))]

    files = group_files(args.clones_root, args.repo_root, targets)

    # Fixed sample per group (same design as the UTMOS22 pass), then ensemble
    # the UTMOSv2 folds by averaging per-file MOS across folds.
    samples = {}
    for g, fs in files.items():
        fs = sorted(fs)
        if not fs:
            continue
        rng_g = random.Random(args.seed)
        samples[g] = fs if len(fs) <= args.sample_per_group else rng_g.sample(fs, args.sample_per_group)

    per_file = {g: np.zeros(len(fl)) for g, fl in samples.items()}
    for fold in folds:
        model = utmosv2.create_model(pretrained=True, fold=fold)
        for g, fl in samples.items():
            per_file[g] += np.array([float(model.predict(input_path=f)) for f in fl])
        print(f"[fold {fold}] scored {sum(len(v) for v in samples.values())} files")
    for g in per_file:
        per_file[g] /= len(folds)

    out = {"meta": {
        "mos_model": f"sarulab-speech/UTMOSv2 (fusion_stage3, {len(folds)}-fold ensemble folds={folds}, s42), 2024",
        "sample_per_group": args.sample_per_group,
        "seed": args.seed,
        "note": "UTMOSv2 re-score of the UTMOS22-strong naturalness sanity "
                "check; DNN-MOS read only as 'clones not degenerate', not a "
                "real-vs-clone naturalness comparison.",
    }, "mos": {}}

    for g in samples:
        s = per_file[g]
        out["mos"][g] = {
            "n": len(s), "mean": float(s.mean()), "std": float(s.std()),
            "median": float(np.median(s)), "min": float(s.min()), "max": float(s.max()),
        }
        print(f"{g:14s} n={len(s):3d} mean={s.mean():.3f} median={np.median(s):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

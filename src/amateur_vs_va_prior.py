"""Quantify the "different prior" (事前分布が違う): how separable are amateur
(Common Voice JA, general public) voices from professional voice actors? High
separability = a strong population prior (professional articulation / 滑舌 + channel)
that a Bayesian/human judge can use to exonerate untrained amateurs.

Crucially we separate channel from articulation: we classify amateur-vs-VA both RAW
and after imposing the SAME channel on both (7 kHz low-pass + MP3-64k round-trip). The
matched-channel residual is the non-channel (articulation/voice-population) part.

Balanced accuracy (chance 50%, class-balanced), speaker-disjoint (1 clip/speaker).

Usage: python -m src.amateur_vs_va_prior --models animeva ecapa
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, torch, soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.embeddings import build_extractor
from src.channel_control import _bandlimit, _codec

CV = "/path/to/va-data/processed/cv"
N_PER = 500  # clips per class (1 per speaker)


def va_files(n):
    # agency (non-freelance) speakers from the embeddings meta; sample 1 processed file each
    d = np.load("output/embeddings/ecapa.npz", allow_pickle=True)
    spk = d["speaker_id"].astype(str); src = d["recording_source"].astype(str)
    agency = sorted(set(spk[~np.char.startswith(src, "freelance:")]))
    rng = np.random.default_rng(0); rng.shuffle(agency)
    files = []
    for s in agency:
        fs = sorted(glob.glob(f"data/processed/{s}/*.wav"))
        if fs:
            files.append(fs[0])
        if len(files) >= n:
            break
    return files


def cv_files(n):
    rng = np.random.default_rng(1)
    by = {}
    for f in sorted(glob.glob(CV + "/*.wav")):
        s = Path(f).stem.split("__")[0]
        by.setdefault(s, f)            # 1 clip per amateur speaker
    spks = list(by); rng.shuffle(spks)
    return [by[s] for s in spks[:n]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    args = ap.parse_args()
    vf, cf = va_files(N_PER), cv_files(N_PER)
    files = vf + cf
    y = np.array([1] * len(vf) + [0] * len(cf))      # 1=VA, 0=amateur
    print(f"VA {len(vf)} / amateur {len(cf)} clips (1 per speaker, speaker-disjoint)")
    out = {}
    for model in args.models:
        ext = build_extractor(model, "cuda")

        def emb(w):
            v = np.asarray(ext.embed(torch.from_numpy(w.astype(np.float32))), np.float32).reshape(-1)
            return v / (np.linalg.norm(v) + 1e-9)

        def sep(transform):
            X = []
            for f in files:
                w, _ = sf.read(f, dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
                X.append(emb(transform(w)))
            X = np.stack(X); X = X - X.mean(0); X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            bal = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean()
            auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
            return float(bal), float(auc)

        raw = sep(lambda w: w)
        matched = sep(lambda w: _codec(_bandlimit(w)))
        out[model] = {"raw_balacc": raw[0], "raw_auc": raw[1],
                      "matched_channel_balacc": matched[0], "matched_channel_auc": matched[1]}
        print(f"\n#### {model} — amateur vs VA separability ####")
        print(f"  RAW:            bal-acc {raw[0]*100:.1f}%  AUC {raw[1]:.3f}  (channel + articulation/population)")
        print(f"  MATCHED-CHANNEL: bal-acc {matched[0]*100:.1f}%  AUC {matched[1]:.3f}  (channel removed → articulation/population residual)")
        print(f"  channel-attributable drop: {(raw[0]-matched[0])*100:.1f} pts")
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/amateur_vs_va_prior.json", "w"), indent=2)
    print("\nwrote output/analysis/amateur_vs_va_prior.json")


if __name__ == "__main__":
    main()

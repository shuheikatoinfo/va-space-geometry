"""Channel-control follow-up for the real-vs-synthetic covariate shift (§4.4).

The headline claim "real and clone audio are ~91% linearly separable" (clone_geometry.py)
could be a CHANNEL artifact rather than genuine synthesis structure: clones carry vocoder
band-limiting / codec / noise-floor signatures that differ from clean studio recordings,
independent of speaker rendering. If so, a channel-normalized detector might recover.

We test this two ways, neither needing clone regeneration:

(1) MATCHED-CHANNEL re-encoding. Apply the SAME channel transform to BOTH real and clone
    waveforms, then re-measure real-vs-synthetic separability with a speaker-DISJOINT split
    (GroupKFold by target). Transforms: band-limit (common 7 kHz low-pass), codec (common
    MP3 64 kbps round-trip), and both. If separability survives matching, the shift is
    intrinsic (in the speaker rendering the encoder sees), not channel. If it collapses
    toward 50%, it was channel.

(2) CHANNEL-INHERITANCE control. kNN-VC concatenates real reference frames, so its output
    channel ~= real channel; TTS (Irodori/GPT-SoVITS) synthesizes a fresh channel. Per-method
    real-vs-synthetic separability therefore isolates channel: if real-vs-kNN-VC is much
    lower than real-vs-TTS, channel drives part of the shift; if even kNN-VC stays high, the
    shift is intrinsic.

Usage: python -m src.channel_control --models animeva ecapa
"""
from __future__ import annotations

import argparse, json, re, subprocess, tempfile, zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch, soundfile as sf
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.clone_probe import va_actor_segments
from src.embeddings import build_extractor

CLONES = {
    # kNN-VC removed (perceptual fidelity too low to be a realistic attack); replaced
    # by Seed-VC v2, a Japanese-capable zero-shot SOTA voice-conversion model.
    "seedvc": "/path/to/va-data/clones/seedvc",
    "irodori": "/path/to/va-data/clones/irodori",
    "gptsovits_v1": "/path/to/va-data/clones/gptsovits_v1",
    "gptsovits_v2": "/path/to/va-data/clones/gptsovits_v2",
    "gptsovits_v2ProPlus": "/path/to/va-data/clones/gptsovits",
    "gptsovits_v3": "/path/to/va-data/clones/gptsovits_v3",
    "gptsovits_v4": "/path/to/va-data/clones/gptsovits_v4",
}
# All remaining systems synthesize a fresh channel (TTS vocoders / Seed-VC's BigVGAN);
# none inherit the source channel, so the channel argument rests on matched re-encoding.
INHERITS_CHANNEL = set()
SR = 16000
MAX_PER_METHOD = 140      # cap clone files per method to bound runtime
REAL_SEGS = 4             # real segments per target
LOWPASS_HZ = 7000         # common band-limit cutoff
MP3_BITRATE = "64k"       # common codec round-trip


# ---- channel transforms (applied identically to real AND clone) ----
def _bandlimit(w):
    sos = butter(8, LOWPASS_HZ / (SR / 2), btype="low", output="sos")
    return sosfiltfilt(sos, w).astype(np.float32)


def _codec(w):
    """MP3 64 kbps encode->decode round-trip via ffmpeg."""
    with tempfile.TemporaryDirectory() as td:
        a, b, c = f"{td}/a.wav", f"{td}/b.mp3", f"{td}/c.wav"
        sf.write(a, w, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-b:a", MP3_BITRATE, b], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", b, "-ar", str(SR), "-ac", "1", c], check=True)
        out, _ = sf.read(c, dtype="float32")
        return (out.mean(1) if out.ndim > 1 else out).astype(np.float32)


TRANSFORMS = {
    "raw": lambda w: w,
    "bandlimit": _bandlimit,
    "codec": _codec,
    "both": lambda w: _codec(_bandlimit(w)),
}


def _load(path):
    w, sr = sf.read(str(path), dtype="float32")
    # embed() does not resample, and _bandlimit's Butterworth design assumes SR
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr} Hz"
    w = w.mean(1) if w.ndim > 1 else w
    return np.asarray(w, np.float32)


def gather_waves(want_targets):
    """Return list of (waveform, method, target). One read per file."""
    rows = []
    for method, d in CLONES.items():
        if not Path(d).exists():
            continue
        files = sorted(Path(d).glob("clone_*__src*.wav"))
        kept = []
        for f in files:
            m = re.match(r"clone_(.+)__src\d+", f.stem)
            if m and m.group(1) in want_targets:
                kept.append((f, m.group(1)))
        # deterministic cap, spread across targets
        if len(kept) > MAX_PER_METHOD:
            kept = kept[:: max(1, len(kept) // MAX_PER_METHOD)][:MAX_PER_METHOD]
        for f, t in kept:
            rows.append((_load(f), method, t))
    real_targets = {t for _, _, t in rows}
    actors = va_actor_segments()
    for t in real_targets:
        for p in actors[t][8:8 + REAL_SEGS]:
            rows.append((_load(p), "real", t))
    return rows


def separability(X, y_real, groups):
    """Speaker-disjoint real-vs-synthetic separability (GroupKFold by target).
    Classes are imbalanced (more clones than real), so we report BALANCED accuracy
    (chance = 50% regardless of imbalance, class_weight='balanced') as the headline,
    plus plain accuracy and the majority-class baseline for reference."""
    X = X - X.mean(0, keepdims=True)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    bal = cross_val_score(clf, X, y_real, groups=groups, cv=gkf, scoring="balanced_accuracy").mean()
    auc = cross_val_score(clf, X, y_real, groups=groups, cv=gkf, scoring="roc_auc").mean()
    # per-class recall (1=real, 0=synth) via out-of-fold predictions
    pred = cross_val_predict(clf, X, y_real, groups=groups, cv=gkf)
    rec_real = recall_score(y_real, pred, pos_label=1)
    rec_synth = recall_score(y_real, pred, pos_label=0)
    majority = max(np.mean(y_real), 1 - np.mean(y_real))
    return float(bal), float(auc), float(rec_real), float(rec_synth), float(majority)


def run(model):
    ext = build_extractor(model, "cuda")
    actors = va_actor_segments()
    want = set(actors.keys())
    waves = gather_waves(want)
    method = np.array([r[1] for r in waves])
    target = np.array([r[2] for r in waves])
    is_real = method == "real"
    print(f"\n#### {model}: {int((~is_real).sum())} clones / {int(is_real.sum())} real, "
          f"{len(set(target))} targets ####")

    res = {"model": model, "n_clone": int((~is_real).sum()), "n_real": int(is_real.sum()),
           "n_targets": int(len(set(target))), "matched_channel": {}, "per_method_raw": {}}

    # (1) matched-channel separability: embed under each transform, re-test
    for cond, fn in TRANSFORMS.items():
        X = np.stack([np.asarray(ext.embed(torch.from_numpy(fn(w))), np.float32).reshape(-1)
                      for w, _, _ in waves])
        y = (is_real).astype(int)            # 1 = real
        bal, auc, rec_real, rec_synth, maj = separability(X, y, target)
        res["matched_channel"][cond] = {"balanced_acc": bal, "roc_auc": auc,
                                        "recall_real": rec_real, "recall_synth": rec_synth,
                                        "majority_baseline": maj}
        print(f"  [{cond:9s}] real-vs-synth: bal-acc {bal*100:5.1f}%  AUC {auc:.3f}  "
              f"(recall real {rec_real*100:.0f}% / synth {rec_synth*100:.0f}%, maj {maj*100:.0f}%)")

    # (2) channel-inheritance control: per-method real-vs-(that method), raw audio, speaker-disjoint
    Xraw = np.stack([np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)
                     for w, _, _ in waves])
    Xraw = Xraw - Xraw.mean(0, keepdims=True)
    Xraw = Xraw / np.clip(np.linalg.norm(Xraw, axis=1, keepdims=True), 1e-9, None)
    print("  per-method real-vs-synth (raw audio, speaker-disjoint):")
    for m in [k for k in CLONES if (method == k).any()]:
        sel = (method == m) | is_real
        if sel.sum() < 20 or len(set(target[sel])) < 4:
            continue
        y = is_real[sel].astype(int)
        g = target[sel]
        gkf = GroupKFold(n_splits=min(5, len(set(g))))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        bal = float(cross_val_score(clf, Xraw[sel], y, groups=g, cv=gkf, scoring="balanced_accuracy").mean())
        maj = float(max(np.mean(y), 1 - np.mean(y)))
        tag = "channel-inheriting" if m in INHERITS_CHANNEL else "fresh-channel TTS"
        res["per_method_raw"][m] = {"balanced_acc": bal, "majority_baseline": maj,
                                    "type": tag, "n_clone": int((method == m).sum())}
        print(f"    {m:22s} bal-acc {bal*100:5.1f}% (maj {maj*100:.0f}%)  ({tag})")

    # (3) cross-synthesizer generalization: does a real-vs-synth detector trained on
    # SOME synthesizers transfer to a HELD-OUT one? Real audio is split 50/50 by target
    # (so real never leaks across train/test); synth is split by synthesizer. If bal-acc
    # stays high the shift is synthesizer-general; if it falls to ~50% it is synth-specific
    # (the ASVspoof/SASV "poor generalization to unseen synthesizers" failure mode).
    print("  cross-synthesizer generalization (train on others + half real, test on held-out synth + other half real):")
    res["cross_synth"] = {}
    real_idx = np.where(is_real)[0]
    # deterministic 50/50 real split by a stable hash of the target string
    real_train_mask = np.array([(zlib.crc32(target[i].encode()) % 2 == 0) for i in real_idx])
    rtr, rte = real_idx[real_train_mask], real_idx[~real_train_mask]
    held = [k for k in ["seedvc", "irodori", "gptsovits_v1", "gptsovits_v4"] if (method == k).any()]
    for S in held:
        tr = np.concatenate([rtr, np.where((~is_real) & (method != S))[0]])
        te = np.concatenate([rte, np.where(method == S)[0]])
        ytr, yte = is_real[tr].astype(int), is_real[te].astype(int)
        if len(set(yte)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xraw[tr], ytr)
        bal = float(balanced_accuracy_score(yte, clf.predict(Xraw[te])))
        res["cross_synth"][S] = {"balanced_acc_heldout": bal, "n_test_synth": int((method == S).sum())}
        print(f"    held-out {S:18s} bal-acc {bal*100:5.1f}%  (trained without {S})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    args = ap.parse_args()
    out = {}
    for m in args.models:
        out[m] = run(m)
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/channel_control.json", "w"), indent=2)
    print("\nwrote output/analysis/channel_control.json")


if __name__ == "__main__":
    main()

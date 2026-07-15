"""Vocoder copy-synthesis control for the real-vs-clone covariate shift (paper §4.4/§7).

The real-vs-clone linear separability (ECAPA ~86%, animeva ~74% balanced acc,
GroupKFold by speaker) could be explained by a NEURAL-VOCODER FINGERPRINT alone:
every clone pipeline ends in a neural vocoder, so a "synthetic" detector might
just detect vocoded audio. Control: copy-synthesize REAL segments through a
neural vocoder (mel analysis -> vocoder resynthesis, no voice conversion) and
measure:

  (a) real vs copy-synth(real) separability (same protocol as channel_control:
      logistic regression on centered+L2-normed embeddings, GroupKFold by
      speaker, balanced accuracy + AUC). If ~= real-vs-clone separability, the
      vocoder fingerprint alone explains the shift; if much lower, the shift is
      more than vocoding.
  (b) classifier trained on real-vs-clone, scored on copy-synth(real): fraction
      classified "synthetic" (with the same speakers' original real segments as
      a control for domain shift).
  (c) speaker-identity sanity check: cosine of copy-synth embedding to its own
      speaker's real centroid vs to other speakers' centroids.

Vocoder: BigVGAN v2 (nvidia/bigvgan_v2_22khz_80band_256x), the vocoder bundled
with Seed-VC v2 — i.e. the same family used by one of the actual clone systems.
16 kHz real segments are resampled to 22.05 kHz, mel-analyzed with BigVGAN's own
mel front-end, resynthesized, and resampled back to 16 kHz mono (embed() does
NOT resample; non-16 kHz input would be silently corrupted).

Usage:
  python -m src.vocoder_copysynth --verify          # 5-file sanity check
  python -m src.vocoder_copysynth --models animeva ecapa
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch, torchaudio, soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import recall_score

from src.clone_probe import va_actor_segments
from src.embeddings import build_extractor

SEEDVC_DIR = "/path/to/va-data/seed-vc"
VOCODER_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
OUT_DIR = Path("/path/to/va-data/copysynth_bigvgan")
SR = 16000

# mirror src/channel_control.py exactly
CLONES = {
    "seedvc": "/path/to/va-data/clones/seedvc",
    "irodori": "/path/to/va-data/clones/irodori",
    "gptsovits_v1": "/path/to/va-data/clones/gptsovits_v1",
    "gptsovits_v2": "/path/to/va-data/clones/gptsovits_v2",
    "gptsovits_v2ProPlus": "/path/to/va-data/clones/gptsovits",
    "gptsovits_v3": "/path/to/va-data/clones/gptsovits_v3",
    "gptsovits_v4": "/path/to/va-data/clones/gptsovits_v4",
}
MAX_PER_METHOD = 140
REAL_SEGS = 4                 # real segments per clone target (indices 8..12)
SEGS_PER_SPEAKER = 4          # copy-synth segments per speaker (indices 8..12)
MAX_SEGMENTS = 800            # total copy-synth budget


# ---------- vocoder ----------
def load_vocoder(device="cuda"):
    sys.path.insert(0, SEEDVC_DIR)
    from huggingface_hub import hf_hub_download
    from modules.bigvgan import bigvgan
    from modules.bigvgan.env import AttrDict
    from modules.bigvgan.meldataset import get_mel_spectrogram
    h = AttrDict(json.load(open(hf_hub_download(VOCODER_REPO, "config.json"))))
    m = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    ckpt = torch.load(hf_hub_download(VOCODER_REPO, "bigvgan_generator.pt"), map_location="cpu")
    m.load_state_dict(ckpt["generator"])
    m.remove_weight_norm()
    m = m.eval().to(device)
    vsr = int(h.sampling_rate)

    def copy_synth(w16: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 -> mel -> BigVGAN -> back to 16 kHz mono float32."""
        with torch.no_grad():
            x = torch.from_numpy(w16).unsqueeze(0)
            x = torchaudio.functional.resample(x, SR, vsr)
            mel = get_mel_spectrogram(x, h).to(device)
            y = m(mel).squeeze(0).cpu().clamp(-1.0, 1.0)
            y = torchaudio.functional.resample(y, vsr, SR)
        return y.squeeze(0).numpy().astype(np.float32)

    return copy_synth


def _load16k(path):
    w, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr} Hz (embed() does not resample)"
    return w.mean(1) if w.ndim > 1 else w


def select_segments():
    """(speaker, real_path) pairs: SEGS_PER_SPEAKER per speaker, capped at MAX_SEGMENTS."""
    actors = va_actor_segments()
    pairs = []
    for s in sorted(actors):
        for p in actors[s][8:8 + SEGS_PER_SPEAKER]:
            pairs.append((s, p))
    if len(pairs) > MAX_SEGMENTS:
        keep_speakers = sorted(actors)[: MAX_SEGMENTS // SEGS_PER_SPEAKER]
        pairs = [(s, p) for s, p in pairs if s in set(keep_speakers)]
    return pairs


def synthesize(pairs, copy_synth):
    """Copy-synthesize each pair -> OUT_DIR; resumable. Returns (speaker, real, cs) list."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (s, p) in enumerate(pairs):
        dst = OUT_DIR / f"cs_{s}__{Path(p).stem}.wav"
        if not dst.exists():
            y = copy_synth(_load16k(p))
            sf.write(str(dst), y, SR)
        out.append((s, p, dst))
        if (i + 1) % 100 == 0:
            print(f"  copy-synth {i + 1}/{len(pairs)}")
    return out


# ---------- clone corpus (mirror channel_control.gather_waves file selection) ----------
def clone_files(want_targets):
    rows = []
    for method, d in CLONES.items():
        if not Path(d).exists():
            continue
        kept = []
        for f in sorted(Path(d).glob("clone_*__src*.wav")):
            m = re.match(r"clone_(.+)__src\d+", f.stem)
            if m and m.group(1) in want_targets:
                kept.append((f, m.group(1)))
        if len(kept) > MAX_PER_METHOD:
            kept = kept[:: max(1, len(kept) // MAX_PER_METHOD)][:MAX_PER_METHOD]
        rows += [(f, method, t) for f, t in kept]
    return rows


# ---------- metrics ----------
def _center_norm(X, mean=None):
    mean = X.mean(0, keepdims=True) if mean is None else mean
    Xc = X - mean
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None), mean


def separability(X, y, groups):
    """Balanced acc / AUC / per-class recall, GroupKFold by speaker (channel_control protocol)."""
    X, _ = _center_norm(X)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    bal = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="balanced_accuracy").mean()
    auc = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean()
    pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf)
    return {"balanced_acc": float(bal), "roc_auc": float(auc),
            "recall_real": float(recall_score(y, pred, pos_label=1)),
            "recall_synth": float(recall_score(y, pred, pos_label=0))}


def run(model, triples, device="cuda"):
    ext = build_extractor(model, device)

    def emb(path):
        return np.asarray(ext.embed(torch.from_numpy(_load16k(path))), np.float32).reshape(-1)

    spk = np.array([s for s, _, _ in triples])
    print(f"\n#### {model}: embedding {len(triples)} real + {len(triples)} copy-synth ####")
    Xreal = np.stack([emb(p) for _, p, _ in triples])
    Xcs = np.stack([emb(c) for _, _, c in triples])

    # (a) real vs copy-synth separability, speaker-disjoint
    X = np.concatenate([Xreal, Xcs])
    y = np.concatenate([np.ones(len(Xreal)), np.zeros(len(Xcs))]).astype(int)  # 1 = real
    g = np.concatenate([spk, spk])
    sep = separability(X, y, g)
    print(f"  (a) real vs copy-synth: bal-acc {sep['balanced_acc']*100:.1f}%  AUC {sep['roc_auc']:.3f}")

    # reference: real-vs-clone under the identical protocol (clone targets only)
    actors = va_actor_segments()
    cl = clone_files(set(actors))
    cl_targets = sorted({t for _, _, t in cl})
    Xcl = np.stack([emb(f) for f, _, _ in cl])
    real_ct = [(t, p) for t in cl_targets for p in actors[t][8:8 + REAL_SEGS]]
    Xrct = np.stack([emb(p) for _, p in real_ct])
    Xrc = np.concatenate([Xrct, Xcl])
    yrc = np.concatenate([np.ones(len(Xrct)), np.zeros(len(Xcl))]).astype(int)
    grc = np.concatenate([np.array([t for t, _ in real_ct]), np.array([t for _, _, t in cl])])
    sep_rc = separability(Xrc, yrc, grc)
    print(f"      real vs clone (ref): bal-acc {sep_rc['balanced_acc']*100:.1f}%  AUC {sep_rc['roc_auc']:.3f}")

    # (b) train real-vs-clone on ALL of Xrc, score copy-synth (and its paired real as control).
    # The classifier's real training side is actors[t][8:8+REAL_SEGS] for every clone target,
    # which is the same slice select_segments() uses — so any copy-synth speaker who is also a
    # clone target would be scored on their own training examples. Evaluate (b) only on
    # speakers OUTSIDE the clone-target set to keep the control out-of-sample.
    Xrc_n, mu = _center_norm(Xrc)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xrc_n, yrc)
    Xcs_n, _ = _center_norm(Xcs, mu)
    Xreal_n, _ = _center_norm(Xreal, mu)
    held = ~np.isin(spk, np.array(cl_targets))
    if held.sum() == 0:
        raise RuntimeError("(b): no copy-synth speaker outside the clone-target set")
    frac_cs_synth = float(np.mean(clf.predict(Xcs_n[held]) == 0))
    frac_real_synth = float(np.mean(clf.predict(Xreal_n[held]) == 0))
    print(f"  (b) real-vs-clone clf on copy-synth (held-out spk only, n={int(held.sum())} segs / "
          f"{len(set(spk[held]))} spk): {frac_cs_synth*100:.1f}% flagged synthetic "
          f"(control: {frac_real_synth*100:.1f}% of the same speakers' REAL segments)")

    # (c) identity preservation: cosine to own real centroid vs other-speaker centroids
    Rn = Xreal / np.clip(np.linalg.norm(Xreal, axis=1, keepdims=True), 1e-9, None)
    Cn = Xcs / np.clip(np.linalg.norm(Xcs, axis=1, keepdims=True), 1e-9, None)
    cents = {}
    for s in sorted(set(spk)):
        c = Rn[spk == s].mean(0)
        cents[s] = c / max(np.linalg.norm(c), 1e-9)
    Cmat = np.stack([cents[s] for s in sorted(cents)])
    order = {s: i for i, s in enumerate(sorted(cents))}
    own = np.array([order[s] for s in spk])
    Scs = Cn @ Cmat.T
    Sre = Rn @ Cmat.T
    n_spk = Cmat.shape[0]
    mask_other = np.ones_like(Scs, bool); mask_other[np.arange(len(spk)), own] = False
    ident = {
        "cos_cs_to_own_centroid": float(Scs[np.arange(len(spk)), own].mean()),
        "cos_cs_to_other_centroids": float(Scs[mask_other].mean()),
        "cos_real_to_own_centroid": float(Sre[np.arange(len(spk)), own].mean()),
        "top1_own_centroid_rate_cs": float(np.mean(Scs.argmax(1) == own)),
        "n_speakers": int(n_spk),
    }
    print(f"  (c) identity: cos(copy-synth, own centroid)={ident['cos_cs_to_own_centroid']:.3f} "
          f"(real={ident['cos_real_to_own_centroid']:.3f}, other-spk={ident['cos_cs_to_other_centroids']:.3f}, "
          f"top-1 own-speaker {ident['top1_own_centroid_rate_cs']*100:.1f}%)")

    return {
        "model": model,
        "n_copysynth": len(Xcs), "n_real": len(Xreal), "n_speakers": int(len(set(spk))),
        "real_vs_copysynth": sep,
        "real_vs_clone_reference": {**sep_rc, "n_clone": len(Xcl), "n_real": len(Xrct),
                                    "n_targets": len(cl_targets)},
        "clone_clf_on_copysynth": {"frac_copysynth_flagged_synthetic": frac_cs_synth,
                                   "frac_real_flagged_synthetic_control": frac_real_synth,
                                   "eval_restricted_to_non_clone_targets": True,
                                   "n_eval_segments": int(held.sum()),
                                   "n_eval_speakers": len(set(spk[held]))},
        "identity_preservation": ident,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--verify", action="store_true", help="copy-synth 5 files and report stats only")
    args = ap.parse_args()

    pairs = select_segments()
    print(f"{len(pairs)} real segments across {len({s for s, _ in pairs})} speakers")
    copy_synth = load_vocoder()

    if args.verify:
        for s, p in pairs[:5]:
            y = copy_synth(_load16k(p))
            dst = OUT_DIR / f"cs_{s}__{Path(p).stem}.wav"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), y, SR)
            info = sf.info(str(dst))
            print(f"  {dst.name}: sr={info.samplerate} dur={info.duration:.2f}s rms={np.sqrt((y**2).mean()):.4f}")
        return

    triples = synthesize(pairs, copy_synth)
    out = {"vocoder": f"BigVGAN v2 ({VOCODER_REPO}, Seed-VC bundled module)",
           "copysynth_dir": str(OUT_DIR), "models": {}}
    for m in args.models:
        out["models"][m] = run(m, triples)
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/vocoder_copysynth.json", "w"), indent=2)
    print("\nwrote output/analysis/vocoder_copysynth.json")


if __name__ == "__main__":
    main()

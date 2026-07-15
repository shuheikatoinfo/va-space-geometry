"""Content-matched real/clone control for the real-vs-clone covariate shift (paper §4.4/§7).

The real-vs-clone linear separability (ECAPA ~86%, animeva ~74% balanced acc,
GroupKFold by speaker) could be partly an artifact of CONTENT mismatch: the
original clone sets speak 6 fixed sentences while the real segments speak
whatever the agency samples contain. Control: for each Irodori clone target,
transcribe held-out REAL segments (positions 8+, never used for enrollment)
with a Japanese ASR, then zero-shot-clone the SAME text with the SAME
Irodori-TTS recipe (same ref_wavs[0:2] reference, same sampler settings) so
real and clone are text-paired. Measure:

  (a) real vs content-MATCHED Irodori clones (paired texts),
  (b) reference: real vs the ORIGINAL content-unmatched Irodori clones on the
      same speakers, same real segments, same protocol
      (logistic regression on centered+L2-normed embeddings, GroupKFold by
      speaker, balanced accuracy + AUC — the channel_control protocol),
  (c) match-quality proxy: re-transcribe the synthesized clones with the same
      ASR and report CER against the target text (upper-bounds ASR+TTS error).

If (a) ~= (b), content mismatch is NOT a driver of the separability; if
(a) << (b), content contributed.

Two stages in two environments (the Irodori runtime needs its own venv):

  # stage 1 — synthesis + ASR (Irodori venv; see clones/irodori_recipe.md)
  HF_HOME=/path/to/va-data/hf_cache \
  /path/to/irodori_venv/bin/python src/content_matched_control.py synth [--limit 3]

  # stage 2 — embeddings + metrics (main project env, from the project dir)
  python -m src.content_matched_control analyze --models ecapa animeva

All wavs are written at 16 kHz mono: embed() does NOT resample, non-16 kHz
input is silently corrupted (see the Seed-VC 22 050 Hz incident).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_DIR = "."
IRODORI_REPO = "/path/to/va-data/Irodori-TTS-repo"
MANIFEST_TARGETS = "/path/to/va-data/clone_targets.json"
ORIG_CLONE_DIR = Path("/path/to/va-data/clones/irodori")
OUT_DIR = Path("/path/to/va-data/clones/irodori_matched")
PAIR_MANIFEST = OUT_DIR / "manifest.json"
SEGMENTS_JSONL = Path(PROJECT_DIR) / "data/processed/segments.jsonl"
OUT_JSON = Path(PROJECT_DIR) / "output/analysis/content_matched_control.json"
SR = 16000
HF_CKPT = "Aratako/Irodori-TTS-500M-v3"
ASR_MODEL = "large-v3"
REAL_SEGS = 4          # held-out real segments per target (positions 8+, as in vocoder_copysynth)
MIN_TEXT_CHARS = 5     # skip segments whose transcript is degenerate
NUM_STEPS = 40         # same sampler settings as the original irodori run
SEED_BASE = 7000       # distinct from the original run's 1234+i


# ---------- shared helpers ----------
def _norm_ja(s: str) -> str:
    """NFKC-normalize and strip whitespace/punctuation for CER."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(c for c in s if not unicodedata.category(c).startswith(("P", "Z", "C")))


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm_ja(ref), _norm_ja(hyp)
    if not r:
        return 1.0 if h else 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def _load16k(path):
    w, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr} Hz (embed() does not resample)"
    return w.mean(1) if w.ndim > 1 else w


def held_out_segments():
    """target_id -> up to REAL_SEGS held-out real segment paths (positions 8+,
    excluding anything used as an enrollment/cloning reference)."""
    targets = json.load(open(MANIFEST_TARGETS))
    by = {}
    for line in open(SEGMENTS_JSONL):
        if '"segment_id"' not in line:
            continue
        r = json.loads(line)
        if r["recording_source"].startswith("freelance:"):  # agency studio only
            continue
        by.setdefault(r["speaker_id"], []).append(r["segment_path"])
    out = []
    for t in targets:
        tid = str(t["target"])
        segs = by.get(tid, [])
        if len(segs) < 8:
            continue
        refs = set(t.get("ref_wavs", []))
        held = [p for p in segs[8:] if p not in refs][:REAL_SEGS]
        if held:
            out.append({"target": tid, "ref_wavs": t.get("ref_wavs", []), "real": held})
    return out


# ---------- stage 1: ASR + content-matched synthesis (Irodori venv) ----------
def cmd_synth(args):
    sys.path.insert(0, IRODORI_REPO)
    import librosa
    import torch
    from faster_whisper import WhisperModel
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

    plan = held_out_segments()
    if args.limit:
        plan = plan[: max(1, (args.limit + REAL_SEGS - 1) // REAL_SEGS)]
        for t in plan:
            t["real"] = t["real"][: args.limit]
    n_seg = sum(len(t["real"]) for t in plan)
    print(f"[info] {len(plan)} targets, {n_seg} held-out real segments", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- ASR of the real segments
    try:
        asr = WhisperModel(ASR_MODEL, device="cuda", compute_type="float16")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] GPU ASR unavailable ({e}); falling back to CPU int8", flush=True)
        asr = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")

    def transcribe(path):
        segs, _ = asr.transcribe(str(path), language="ja", beam_size=5, vad_filter=False)
        return "".join(s.text for s in segs).strip()

    rows = []
    for t in plan:
        for k, rel in enumerate(t["real"]):
            full = os.path.join(PROJECT_DIR, rel)
            text = transcribe(full)
            rows.append({"target": t["target"], "idx": k, "real_path": full,
                         "ref_wavs": t["ref_wavs"], "text": text})
            print(f"[asr] {t['target']} seg{k}: {text[:60]}", flush=True)
    kept = [r for r in rows if len(_norm_ja(r["text"])) >= MIN_TEXT_CHARS]
    print(f"[info] ASR done: {len(kept)}/{len(rows)} segments have usable transcripts", flush=True)

    # -- Irodori runtime (identical recipe to clones/run_irodori_clones.py)
    from huggingface_hub import hf_hub_download
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=hf_hub_download(repo_id=HF_CKPT, filename="model.safetensors"),
        model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision="bf16", codec_device="cuda", codec_precision="fp32"))
    print("[info] irodori runtime loaded", flush=True)

    def build_reference(ref_paths, tmpdir):
        chunks, sr_out = [], None
        for rp in ref_paths[:2]:
            full = os.path.join(PROJECT_DIR, rp)
            if not os.path.isfile(full):
                continue
            data, sr = sf.read(full, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            sr_out = sr
            chunks.append(data)
        if not chunks:
            return None
        out = os.path.join(tmpdir, "ref.wav")
        sf.write(out, np.concatenate(chunks), sr_out)
        return out

    manifest, failures = [], []
    ref_cache_target = None
    with tempfile.TemporaryDirectory() as tmp:
        for r in kept:
            if ref_cache_target != r["target"]:
                ref = build_reference(r["ref_wavs"], tmp)
                ref_cache_target = r["target"]
            if ref is None:
                failures.append((r["target"], r["idx"], "no ref"))
                continue
            dst = OUT_DIR / f"clone_{r['target']}__seg{r['idx']:02d}.wav"
            try:
                if not dst.exists():
                    res = runtime.synthesize(
                        SamplingRequest(text=r["text"], ref_wav=ref,
                                        num_steps=NUM_STEPS, seed=SEED_BASE + r["idx"]),
                        log_fn=None)
                    a = res.audio.detach().to("cpu", torch.float32)
                    a = (a.squeeze(0) if a.shape[0] == 1 else a.mean(0)).numpy()
                    in_sr = int(res.sample_rate)
                    if in_sr != SR:
                        a = librosa.resample(a, orig_sr=in_sr, target_sr=SR)
                    peak = float(np.max(np.abs(a))) if a.size else 0.0
                    if peak > 1.0:
                        a = a / peak
                    sf.write(str(dst), a, SR)
                hyp = transcribe(dst)
                cer = _cer(r["text"], hyp)
                manifest.append({**r, "clone_path": str(dst), "clone_asr": hyp, "cer": cer})
                print(f"[ok] {r['target']} seg{r['idx']:02d} cer={cer:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                failures.append((r["target"], r["idx"], repr(e)))
                print(f"[fail] {r['target']} seg{r['idx']:02d}: {e}", flush=True)

    meta = {"asr_model": ASR_MODEL, "tts": HF_CKPT, "num_steps": NUM_STEPS,
            "seed_base": SEED_BASE, "n_targets": len({m["target"] for m in manifest}),
            "n_pairs": len(manifest), "n_asr_skipped": len(rows) - len(kept),
            "failures": failures}
    json.dump({"meta": meta, "pairs": manifest},
              open(PAIR_MANIFEST, "w"), ensure_ascii=False, indent=1)
    cers = [m["cer"] for m in manifest]
    print(f"\n==== SYNTH SUMMARY ====\npairs: {len(manifest)} "
          f"targets: {meta['n_targets']} failures: {len(failures)}", flush=True)
    if cers:
        print(f"CER mean={np.mean(cers):.3f} median={np.median(cers):.3f}", flush=True)
    print(f"manifest: {PAIR_MANIFEST}", flush=True)


# ---------- stage 2: embeddings + metrics (main env) ----------
def _center_norm(X, mean=None):
    mean = X.mean(0, keepdims=True) if mean is None else mean
    Xc = X - mean
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None), mean


def separability(X, y, groups):
    """channel_control/vocoder_copysynth protocol: logistic regression on
    centered+L2-normed embeddings, GroupKFold by speaker."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import recall_score
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
    X, _ = _center_norm(X)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    bal = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="balanced_accuracy").mean()
    auc = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean()
    pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf)
    return {"balanced_acc": float(bal), "roc_auc": float(auc),
            "recall_real": float(recall_score(y, pred, pos_label=1)),
            "recall_synth": float(recall_score(y, pred, pos_label=0))}


def cmd_analyze(args):
    import torch
    from src.embeddings import build_extractor

    man = json.load(open(PAIR_MANIFEST))
    pairs = man["pairs"]
    targets = sorted({p["target"] for p in pairs})
    tset = set(targets)

    # original (content-unmatched) irodori clones for the same targets
    orig = []
    for f in sorted(ORIG_CLONE_DIR.glob("clone_*__src*.wav")):
        m = re.match(r"clone_(.+)__src\d+", f.stem)
        if m and m.group(1) in tset:
            orig.append((m.group(1), f))
    print(f"{len(pairs)} matched pairs, {len(orig)} original clones, {len(targets)} targets")

    out = {"design": "content-matched real/clone control (Irodori-TTS zero-shot)",
           "asr_model": man["meta"]["asr_model"], "tts": man["meta"]["tts"],
           "n_targets": len(targets), "n_pairs": len(pairs),
           "n_original_clones": len(orig),
           "cer": {"mean": float(np.mean([p["cer"] for p in pairs])),
                   "median": float(np.median([p["cer"] for p in pairs])),
                   "frac_le_0.1": float(np.mean([p["cer"] <= 0.1 for p in pairs])),
                   "frac_le_0.3": float(np.mean([p["cer"] <= 0.3 for p in pairs]))},
           "models": {}}

    for model in args.models:
        ext = build_extractor(model, args.device)

        def emb(path):
            return np.asarray(ext.embed(torch.from_numpy(_load16k(path))),
                              np.float32).reshape(-1)

        print(f"\n#### {model} ####")
        Xreal = np.stack([emb(p["real_path"]) for p in pairs])
        Xmat = np.stack([emb(p["clone_path"]) for p in pairs])
        Xorig = np.stack([emb(f) for _, f in orig])
        spk = np.array([p["target"] for p in pairs])
        spk_o = np.array([t for t, _ in orig])

        # (a) real vs content-matched clones (paired texts)
        Xa = np.concatenate([Xreal, Xmat])
        ya = np.concatenate([np.ones(len(Xreal)), np.zeros(len(Xmat))]).astype(int)
        ga = np.concatenate([spk, spk])
        sep_a = separability(Xa, ya, ga)
        print(f"  (a) real vs matched clones:   bal-acc {sep_a['balanced_acc']*100:.1f}%  "
              f"AUC {sep_a['roc_auc']:.3f}")

        # (b) real vs original content-UNmatched clones, same speakers/protocol
        Xb = np.concatenate([Xreal, Xorig])
        yb = np.concatenate([np.ones(len(Xreal)), np.zeros(len(Xorig))]).astype(int)
        gb = np.concatenate([spk, spk_o])
        sep_b = separability(Xb, yb, gb)
        print(f"  (b) real vs original clones:  bal-acc {sep_b['balanced_acc']*100:.1f}%  "
              f"AUC {sep_b['roc_auc']:.3f}")

        out["models"][model] = {
            "real_vs_matched_clones": {**sep_a, "n_real": len(Xreal), "n_clone": len(Xmat)},
            "real_vs_original_clones_reference": {**sep_b, "n_real": len(Xreal),
                                                  "n_clone": len(Xorig)},
        }
        del ext

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("synth")
    ps.add_argument("--limit", type=int, default=0,
                    help="verify mode: only this many segments total")
    pa = sub.add_parser("analyze")
    pa.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    pa.add_argument("--device", default="cuda")
    args = ap.parse_args()
    (cmd_synth if args.cmd == "synth" else cmd_analyze)(args)


if __name__ == "__main__":
    main()

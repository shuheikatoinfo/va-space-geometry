"""Content-matched real/clone control for Seed-VC and GPT-SoVITS v4 (extends
src/content_matched_control.py, which covers Irodori-TTS only; disclosed in §7).

Design (mirrors content_matched_control.py):

  Seed-VC (voice conversion, no ASR needed): for each of the 120 Seed-VC
  targets, take up to 4 held-out real segments (positions 8+, reference audio
  excluded) and SELF-CONVERT each segment through Seed-VC v2 using the segment
  itself as source and the target's own standard concatenated reference as the
  voice prompt (same make_ref recipe as clones/run_seedvc.py). The output
  speaks the same text with the same prosody as the real segment — content AND
  prosody are matched. If separability persists here, the covariate-shift
  reading is even stronger than for the text-only-matched Irodori control.

  GPT-SoVITS v4 (zero-shot TTS, 40 targets): ASR the held-out real segments
  with faster-whisper large-v3 (as in content_matched_control.py) and
  synthesize the same text with the same v4 recipe as
  clones/run_gptsovits_ver.py (same ref_wavs[0], same prompt-transcription).

Metrics per synthesizer (encoders ecapa + animeva; logistic regression on
centered+L2-normed embeddings, GroupKFold by speaker, balanced acc + AUC):
  (a)  real vs content-matched clones,
  (b)  reference: real vs the ORIGINAL content-unmatched clones of the same
       synthesizer, same speakers, same protocol.

Stages / environments:

  # stage 1a — Seed-VC self-conversion (seedvc venv; script chdirs to the repo)
  /path/to/seedvc_venv/bin/python src/content_matched_seedvc.py synth-seedvc [--limit 3]

  # stage 1b — GPT-SoVITS v4 (gptsovits venv)
  HF_HOME=/path/to/va-data/hf_cache \
  /path/to/gptsovits_venv/bin/python src/content_matched_seedvc.py synth-gsv [--limit 3]

  # stage 2 — embeddings + metrics (main project env, from the project dir)
  python -m src.content_matched_seedvc analyze --models ecapa animeva

All wavs are written at 16 kHz mono: embed() does NOT resample.
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
MANIFEST_TARGETS = "/path/to/va-data/clone_targets.json"
SEGMENTS_JSONL = Path(PROJECT_DIR) / "data/processed/segments.jsonl"
SR = 16000
REAL_SEGS = 4
MIN_TEXT_CHARS = 5
ASR_MODEL = "large-v3"

SEEDVC_REPO = "/path/to/va-data/seed-vc"
SEEDVC_ORIG = Path("/path/to/va-data/clones/seedvc")
SEEDVC_OUT = Path("/path/to/va-data/clones/seedvc_matched")
SEEDVC_MANIFEST = SEEDVC_OUT / "manifest.json"
SEEDVC_STEPS = 30  # same as run_seedvc.py

GSV_REPO = "/path/to/GPT-SoVITS"
GSV_MODELS = "/path/to/va-data/gptsovits_models"
GSV_ORIG = Path("/path/to/va-data/clones/gptsovits_v4")
GSV_OUT = Path("/path/to/va-data/clones/gptsovits_matched")
GSV_MANIFEST = GSV_OUT / "manifest.json"
GSV_TARGETS = 40  # same subset as the original gptsovits_v4 run (first 40)

OUT_JSON = Path(PROJECT_DIR) / "output/analysis/content_matched_seedvc.json"


# ---------- shared helpers (identical to content_matched_control.py) ----------
def _norm_ja(s: str) -> str:
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


def held_out_segments(n_targets=None):
    """target_id -> up to REAL_SEGS held-out real segment paths (positions 8+,
    excluding anything used as an enrollment/cloning reference). Same logic as
    content_matched_control.held_out_segments."""
    targets = json.load(open(MANIFEST_TARGETS))
    if n_targets:
        targets = targets[:n_targets]
    by = {}
    for line in open(SEGMENTS_JSONL):
        if '"segment_id"' not in line:
            continue
        r = json.loads(line)
        if r["recording_source"].startswith("freelance:"):
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


def _apply_limit(plan, limit):
    if limit:
        plan = plan[: max(1, (limit + REAL_SEGS - 1) // REAL_SEGS)]
        for t in plan:
            t["real"] = t["real"][:limit]
    return plan


def _to_16k_mono(audio, sr):
    audio = np.asarray(audio, np.float32)
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


# ---------- stage 1a: Seed-VC self-conversion (seedvc venv) ----------
def cmd_synth_seedvc(args):
    os.chdir(SEEDVC_REPO)
    sys.path.insert(0, SEEDVC_REPO)
    from inference_v2 import convert_voice_v2  # noqa: E402 (needs repo cwd)

    class A:  # same quality-focused defaults as clones/run_seedvc.py
        ar_checkpoint_path = None
        cfm_checkpoint_path = None
        compile = False
        diffusion_steps = SEEDVC_STEPS
        length_adjust = 1.0
        intelligibility_cfg_rate = 0.7
        similarity_cfg_rate = 0.7
        top_p = 0.9
        temperature = 1.0
        repetition_penalty = 1.0
        convert_style = False
        anonymization_only = False

    def make_ref(ref_wavs, tmpdir, max_s=12.0):  # identical to run_seedvc.py
        chunks, sr0, total = [], None, 0.0
        for rw in ref_wavs:
            p = Path(PROJECT_DIR) / rw
            if not p.exists():
                continue
            w, sr = sf.read(str(p), dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            sr0 = sr0 or sr
            if sr != sr0:
                continue
            chunks.append(w)
            total += len(w) / sr
            if total >= max_s:
                break
        if not chunks:
            return None
        rp = Path(tmpdir) / "ref.wav"
        sf.write(str(rp), np.concatenate(chunks), sr0)
        return str(rp)

    plan = _apply_limit(held_out_segments(), args.limit)
    n_seg = sum(len(t["real"]) for t in plan)
    print(f"[info] {len(plan)} targets, {n_seg} held-out real segments", flush=True)
    SEEDVC_OUT.mkdir(parents=True, exist_ok=True)

    manifest, failures = [], []
    for ti, t in enumerate(plan):
        with tempfile.TemporaryDirectory() as td:
            ref = make_ref(t["ref_wavs"], td)
            if ref is None:
                failures.append((t["target"], -1, "no ref"))
                continue
            for k, rel in enumerate(t["real"]):
                src = os.path.join(PROJECT_DIR, rel)
                dst = SEEDVC_OUT / f"clone_{t['target']}__seg{k:02d}.wav"
                try:
                    if not dst.exists():
                        sr, audio = convert_voice_v2(src, ref, A)
                        sf.write(str(dst), _to_16k_mono(audio, sr), SR)
                    manifest.append({"target": t["target"], "idx": k,
                                     "real_path": src, "clone_path": str(dst)})
                except Exception as e:  # noqa: BLE001
                    failures.append((t["target"], k, repr(e)))
                    print(f"[fail] {t['target']} seg{k}: {e}", flush=True)
        if (ti + 1) % 10 == 0:
            print(f"  ...{ti+1}/{len(plan)} targets, {len(manifest)} clones", flush=True)

    meta = {"method": "seed-vc v2 self-conversion (source = held-out real segment, "
                      "prompt = target's standard concatenated reference)",
            "diffusion_steps": SEEDVC_STEPS,
            "n_targets": len({m["target"] for m in manifest}),
            "n_pairs": len(manifest), "failures": failures}
    json.dump({"meta": meta, "pairs": manifest},
              open(SEEDVC_MANIFEST, "w"), ensure_ascii=False, indent=1)
    print(f"\n==== SEED-VC SYNTH SUMMARY ====\npairs: {len(manifest)} "
          f"targets: {meta['n_targets']} failures: {len(failures)}\n"
          f"manifest: {SEEDVC_MANIFEST}", flush=True)


# ---------- stage 1b: GPT-SoVITS v4 content-matched synthesis (gptsovits venv) ----------
def cmd_synth_gsv(args):
    os.chdir(GSV_REPO)
    sys.path.insert(0, GSV_REPO)
    sys.path.insert(0, os.path.join(GSV_REPO, "GPT_SoVITS"))
    os.environ.setdefault("HF_HOME", "/path/to/va-data/hf_cache")
    ft = "/path/to/va-data/fast_langdetect"
    os.makedirs(ft, exist_ok=True)
    os.makedirs(os.path.join(GSV_REPO, "GPT_SoVITS/pretrained_models/fast_langdetect"),
                exist_ok=True)
    os.environ["FTLANG_CACHE"] = ft
    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from faster_whisper import WhisperModel

    plan = _apply_limit(held_out_segments(GSV_TARGETS), args.limit)
    n_seg = sum(len(t["real"]) for t in plan)
    print(f"[info] {len(plan)} targets, {n_seg} held-out real segments", flush=True)
    GSV_OUT.mkdir(parents=True, exist_ok=True)

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
    print(f"[info] ASR done: {len(kept)}/{len(rows)} usable transcripts", flush=True)

    # v4 recipe identical to clones/run_gptsovits_ver.py
    cfg = TTS_Config({"custom": {
        "device": "cuda", "is_half": True, "version": "v4",
        "t2s_weights_path": f"{GSV_MODELS}/s1v3.ckpt",
        "vits_weights_path": f"{GSV_MODELS}/gsv-v4-pretrained/s2Gv4.pth",
        "bert_base_path": f"{GSV_MODELS}/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": f"{GSV_MODELS}/chinese-hubert-base"}})
    tts = TTS(cfg)
    print("[info] gpt-sovits v4 runtime loaded", flush=True)

    manifest, failures = [], []
    ptext_cache = {}
    for r in kept:
        tid = r["target"]
        ref = os.path.join(PROJECT_DIR, r["ref_wavs"][0])
        if tid not in ptext_cache:
            try:
                ptext_cache[tid] = transcribe(ref) or "こんにちは。"
            except Exception:  # noqa: BLE001
                ptext_cache[tid] = "こんにちは。"
        dst = GSV_OUT / f"clone_{tid}__seg{r['idx']:02d}.wav"
        try:
            if not dst.exists():
                sr, audio = next(tts.run({
                    "text": r["text"], "text_lang": "ja", "ref_audio_path": ref,
                    "prompt_text": ptext_cache[tid], "prompt_lang": "ja",
                    "batch_size": 1}))
                sf.write(str(dst), _to_16k_mono(audio, sr), SR, subtype="PCM_16")
            hyp = transcribe(dst)
            cer = _cer(r["text"], hyp)
            manifest.append({**r, "clone_path": str(dst), "clone_asr": hyp, "cer": cer})
            print(f"[ok] {tid} seg{r['idx']:02d} cer={cer:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append((tid, r["idx"], repr(e)))
            print(f"[fail] {tid} seg{r['idx']:02d}: {repr(e)[:150]}", flush=True)

    meta = {"method": "gpt-sovits v4 zero-shot, same recipe as run_gptsovits_ver.py, "
                      "text = faster-whisper large-v3 transcript of held-out real segment",
            "asr_model": ASR_MODEL,
            "n_targets": len({m["target"] for m in manifest}),
            "n_pairs": len(manifest), "n_asr_skipped": len(rows) - len(kept),
            "failures": failures}
    json.dump({"meta": meta, "pairs": manifest},
              open(GSV_MANIFEST, "w"), ensure_ascii=False, indent=1)
    cers = [m["cer"] for m in manifest]
    print(f"\n==== GSV SYNTH SUMMARY ====\npairs: {len(manifest)} "
          f"targets: {meta['n_targets']} failures: {len(failures)}", flush=True)
    if cers:
        print(f"CER mean={np.mean(cers):.3f} median={np.median(cers):.3f}", flush=True)
    print(f"manifest: {GSV_MANIFEST}", flush=True)


# ---------- stage 2: embeddings + metrics (main env) ----------
def _center_norm(X, mean=None):
    mean = X.mean(0, keepdims=True) if mean is None else mean
    Xc = X - mean
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None), mean


def separability(X, y, groups):
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


def _analyze_one(name, manifest_path, orig_dir, models, device, out):
    import torch
    from src.embeddings import build_extractor

    man = json.load(open(manifest_path))
    pairs = man["pairs"]
    targets = sorted({p["target"] for p in pairs})
    tset = set(targets)

    orig = []
    for f in sorted(orig_dir.glob("clone_*__src*.wav")):
        m = re.match(r"clone_(.+)__src\d+", f.stem)
        if m and m.group(1) in tset:
            orig.append((m.group(1), f))
    print(f"\n===== {name}: {len(pairs)} matched pairs, {len(orig)} original clones, "
          f"{len(targets)} targets =====")

    entry = {"meta": man["meta"], "n_targets": len(targets), "n_pairs": len(pairs),
             "n_original_clones": len(orig), "models": {}}
    if pairs and "cer" in pairs[0]:
        entry["cer"] = {"mean": float(np.mean([p["cer"] for p in pairs])),
                        "median": float(np.median([p["cer"] for p in pairs])),
                        "frac_le_0.1": float(np.mean([p["cer"] <= 0.1 for p in pairs])),
                        "frac_le_0.3": float(np.mean([p["cer"] <= 0.3 for p in pairs]))}

    for model in models:
        ext = build_extractor(model, device)

        def emb(path):
            return np.asarray(ext.embed(torch.from_numpy(_load16k(path))),
                              np.float32).reshape(-1)

        print(f"\n#### {model} ####")
        Xreal = np.stack([emb(p["real_path"]) for p in pairs])
        Xmat = np.stack([emb(p["clone_path"]) for p in pairs])
        Xorig = np.stack([emb(f) for _, f in orig])
        spk = np.array([p["target"] for p in pairs])
        spk_o = np.array([t for t, _ in orig])

        Xa = np.concatenate([Xreal, Xmat])
        ya = np.concatenate([np.ones(len(Xreal)), np.zeros(len(Xmat))]).astype(int)
        sep_a = separability(Xa, ya, np.concatenate([spk, spk]))
        print(f"  (a) real vs matched clones:   bal-acc {sep_a['balanced_acc']*100:.1f}%  "
              f"AUC {sep_a['roc_auc']:.3f}")

        Xb = np.concatenate([Xreal, Xorig])
        yb = np.concatenate([np.ones(len(Xreal)), np.zeros(len(Xorig))]).astype(int)
        sep_b = separability(Xb, yb, np.concatenate([spk, spk_o]))
        print(f"  (b) real vs original clones:  bal-acc {sep_b['balanced_acc']*100:.1f}%  "
              f"AUC {sep_b['roc_auc']:.3f}")

        entry["models"][model] = {
            "real_vs_matched_clones": {**sep_a, "n_real": len(Xreal), "n_clone": len(Xmat)},
            "real_vs_original_clones_reference": {**sep_b, "n_real": len(Xreal),
                                                  "n_clone": len(Xorig)},
        }
        del ext
    out[name] = entry


def cmd_analyze(args):
    out = {"design": "content-matched real/clone control extended to Seed-VC "
                     "(self-conversion: content AND prosody matched) and GPT-SoVITS v4 "
                     "(ASR text-matched); protocol identical to content_matched_control.py",
           "notes": [
               "Seed-VC control uses self-conversion: the held-out real segment itself is "
               "the VC source, so content and prosody are both matched. If separability "
               "persists, the covariate-shift reading is even stronger than for the "
               "text-only-matched Irodori control.",
               "Reference (b) uses the original content-unmatched clones of the same "
               "synthesizer restricted to the same speakers, same real segments, same "
               "separability protocol (logreg on centered+L2-normed embeddings, "
               "GroupKFold by speaker).",
           ],
           "caveats": [
               "Seed-VC self-conversion also removes prosody/style mismatch, so (a) vs (b) "
               "for Seed-VC bounds content+prosody jointly, not content alone.",
               "GPT-SoVITS matching is approximate: ASR (large-v3) + TTS errors upper-bound "
               "the mismatch (see CER block); prompt transcription uses the same "
               "run_gptsovits_ver.py recipe (whisper of ref_wavs[0]).",
               "Within-synthesizer comparisons only; numbers are not directly comparable "
               "to the all-synthesizer separability in the main text.",
           ],
           "synthesizers": {}}
    if SEEDVC_MANIFEST.exists():
        _analyze_one("seedvc", SEEDVC_MANIFEST, SEEDVC_ORIG, args.models, args.device,
                     out["synthesizers"])
    else:
        print("[warn] no seed-vc manifest; skipping")
    if GSV_MANIFEST.exists():
        _analyze_one("gptsovits_v4", GSV_MANIFEST, GSV_ORIG, args.models, args.device,
                     out["synthesizers"])
    else:
        print("[warn] no gpt-sovits manifest; skipping")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("synth-seedvc", "synth-gsv"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=0,
                       help="verify mode: only this many segments per target-slice")
    pa = sub.add_parser("analyze")
    pa.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    pa.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"synth-seedvc": cmd_synth_seedvc, "synth-gsv": cmd_synth_gsv,
     "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

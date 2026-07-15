"""R3: synthetic-clone probe (defensive). Does a voice clone of actor T land where
a threshold-based unauthorized-generation detector would (a) correctly flag T,
(b) falsely implicate a near-neighbor actor, and (c) sit inside the thin-margin
critical region?

Method (kNN-VC voice conversion): for each target actor T we build a matching set
from T's real studio segments and convert several *source* utterances (JVS speech
— natural Japanese, no VA content) into T's timbre. Each clone is embedded and
scored against a gallery of per-actor centroids built from real audio.

For every clone of T we report:
  - sim_to_T          : cosine to T's real centroid
  - nearest_actor      : the real actor the clone is closest to (== T -> correct
                         attribution; != T -> FALSE attribution / wrongful accusation)
  - attribution_margin : sim_to_T - max sim to any *other* actor (small/negative =
                         the detector cannot cleanly separate T from a neighbor)
  - detectable@thr     : whether sim_to_T exceeds an EER operating threshold

Outputs: output/analysis/clone_probe.json and the generated clones under
         /path/to/va-data/clones.

Usage:
    python -m src.clone_probe --n-targets 40 --n-sources 6 --models animeva ecapa
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

from src.embeddings import build_extractor

CLONE_DIR = Path("/path/to/va-data/clones/knnvc")
JVS_SEG = Path("/path/to/va-data/processed/jvs")
OUT = Path("output/analysis/clone_probe.json")


def va_actor_segments():
    by = defaultdict(list)
    for l in open("data/processed/segments.jsonl"):
        if '"segment_id"' not in l:
            continue
        r = json.loads(l)
        if not r["recording_source"].startswith("freelance:"):  # agency studio only
            by[r["speaker_id"]].append(r["segment_path"])
    return {s: p for s, p in by.items() if len(p) >= 8}


def safe_matching_set(knn, refs, want=6):
    """Build the kNN-VC matching set one reference at a time so a single bad
    segment (e.g. near-silent audio that VAD trims to 0 frames, which crashes
    torchaudio.Vad) is skipped instead of killing the whole target/run."""
    feats = []
    for p in refs:
        if len(feats) >= want:
            break
        try:
            f = knn.get_features(str(p))
            if f is not None and f.shape[0] > 0:
                feats.append(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip ref {Path(p).name}: {exc}")
    if not feats:
        return None
    return torch.concat(feats, dim=0).cpu()


def generate_clones(targets, n_sources):
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                         pretrained=True, device="cuda")
    # Keep only sources with enough audio for WavLM (>= ~2s).
    sources = []
    for p in sorted(JVS_SEG.glob("*.wav")):
        info = sf.info(str(p))
        if info.frames >= 2 * info.samplerate:
            sources.append(p)
        if len(sources) >= n_sources:
            break
    manifest = []
    for ti, (T, refs) in enumerate(targets.items()):
        mset = safe_matching_set(knn, refs)
        if mset is None:
            print(f"  skip target {T}: no usable reference segments")
            continue
        for si, src in enumerate(sources):
            outp = CLONE_DIR / f"clone_{T}__src{si:02d}.wav"
            try:
                if not outp.exists():
                    q = knn.get_features(str(src))
                    out = knn.match(q, mset, topk=4)
                    if out.numel() < 16000:
                        continue
                    torchaudio.save(str(outp), out[None].cpu(), 16000)
                manifest.append({"target": T, "clone_path": str(outp)})
            except Exception as exc:  # noqa: BLE001
                print(f"  skip clone {T} src{si}: {exc}")
        if ti % 10 == 0:
            print(f"  cloned {ti+1}/{len(targets)} targets")
    return manifest


def gallery_and_score(model, targets, manifest):
    ext = build_extractor(model, "cuda")

    def emb_file(p):
        w, sr = sf.read(p, dtype="float32")
        assert sr == 16000, f"{p}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
        if w.ndim > 1:
            w = w.mean(1)
        return np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)

    # Per-actor centroid from real segments (L2-normalized mean), centered later.
    actor_ids = list(targets.keys())
    cents = []
    for T in actor_ids:
        es = np.stack([emb_file(p) for p in targets[T][:8]])
        es = es / np.clip(np.linalg.norm(es, axis=1, keepdims=True), 1e-9, None)
        cents.append(es.mean(0))
    C = np.stack(cents)
    mean = C.mean(0, keepdims=True)
    Cn = (C - mean) / np.clip(np.linalg.norm(C - mean, axis=1, keepdims=True), 1e-9, None)
    idx_of = {T: i for i, T in enumerate(actor_ids)}

    # Same-speaker (real vs real) and impostor scores to set an EER threshold.
    gen, imp = [], []
    for T in actor_ids:
        es = np.stack([emb_file(p) for p in targets[T][8:12]]) if len(targets[T]) > 8 else None
        if es is None:
            continue
        es = (es - mean) / np.clip(np.linalg.norm(es - mean, axis=1, keepdims=True), 1e-9, None)
        sims = es @ Cn.T
        gen += list(sims[:, idx_of[T]])
        imp += list(np.delete(sims, idx_of[T], axis=1).max(1))
    thr = eer_threshold(np.array(gen), np.array(imp))

    rows = []
    for mrec in manifest:
        T = mrec["target"]
        v = emb_file(mrec["clone_path"])
        v = (v - mean.squeeze()) / max(np.linalg.norm(v - mean.squeeze()), 1e-9)
        sims = Cn @ v
        sim_T = float(sims[idx_of[T]])
        others = np.delete(sims, idx_of[T])
        best_other = float(others.max())
        nearest = actor_ids[int(np.argmax(sims))]
        rows.append({"target": T, "sim_to_T": sim_T, "best_other": best_other,
                     "attribution_margin": sim_T - best_other,
                     "correct_attribution": nearest == T,
                     "detectable": sim_T >= thr,
                     "collateral_neighbors": int((others >= thr).sum())})
    return rows, thr


def eer_threshold(gen, imp):
    s = np.concatenate([gen, imp]); lb = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); lb = lb[o]; s = s[o]
    P, N = lb.sum(), len(lb) - lb.sum()
    tp = np.cumsum(lb); fp = np.cumsum(1 - lb)
    fnr = 1 - tp / P; fpr = fp / N
    i = np.argmin(np.abs(fnr - fpr))
    return float(s[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    args = ap.parse_args()

    alla = va_actor_segments()
    rng = np.random.default_rng(0)
    chosen = list(rng.permutation(list(alla.keys())))[:args.n_targets]
    targets = {T: alla[T] for T in chosen}
    print(f"{len(targets)} target actors; generating kNN-VC clones...")
    manifest = generate_clones(targets, args.n_sources)
    print(f"{len(manifest)} clones generated")

    results = {}
    for model in args.models:
        rows, thr = gallery_and_score(model, targets, manifest)
        n = len(rows)
        results[model] = {
            "threshold_eer": thr,
            "n_clones": n,
            "detectable_rate": float(np.mean([r["detectable"] for r in rows])),
            "correct_attribution_rate": float(np.mean([r["correct_attribution"] for r in rows])),
            "false_attribution_rate": float(np.mean([not r["correct_attribution"] for r in rows])),
            "mean_attribution_margin": float(np.mean([r["attribution_margin"] for r in rows])),
            "thin_margin_rate(<0.05)": float(np.mean([abs(r["attribution_margin"]) < 0.05 for r in rows])),
            "mean_collateral_neighbors": float(np.mean([r["collateral_neighbors"] for r in rows])),
        }
        r = results[model]
        print(f"\n## {model} (kNN-VC clones, {n}) ##")
        print(f"  detectable@EER-thr:        {r['detectable_rate']*100:.1f}%")
        print(f"  CORRECT attribution:       {r['correct_attribution_rate']*100:.1f}%")
        print(f"  FALSE attribution:         {r['false_attribution_rate']*100:.1f}%  (clone nearest a DIFFERENT actor)")
        print(f"  mean attribution margin:   {r['mean_attribution_margin']:.3f}")
        print(f"  thin-margin clones(<0.05): {r['thin_margin_rate(<0.05)']*100:.1f}%")
        print(f"  mean collateral neighbors: {r['mean_collateral_neighbors']:.2f} (innocent actors also over threshold)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

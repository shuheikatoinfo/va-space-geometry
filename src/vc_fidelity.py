"""Independent fidelity check for the Seed-VC clones, backing the paper's claim
that Seed-VC is a genuinely high-fidelity attack:
  (a) ECAPA raw-cosine of each clone to its TARGET actor's real centroid,
      vs the real same-speaker baseline (held-out real segment -> own centroid);
  (b) Resemblyzer (the VC-standard speaker encoder, independent of the ECAPA/
      animeva encoders used for attribution) similarity of each clone to the
      target, vs its real same-speaker baseline.

All cosines are RAW (uncentered, L2-normed), matching how the fidelity numbers
are reported in the manuscript (0.72 to target vs 0.63 real same-speaker on
ECAPA; 0.80-0.85 on Resemblyzer).

Usage: python -m src.vc_fidelity --clone-dir /path/to/va-data/clones/seedvc
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np, torch, soundfile as sf

from src.centered_gallery import agency_actor_segments
from src.embeddings import build_extractor


def ecapa_emb(ext, p):
    w, sr = sf.read(str(p), dtype="float32")
    assert sr == 16000, f"{p}: expected 16 kHz, got {sr}"
    if w.ndim > 1:
        w = w.mean(1)
    v = np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dir", default="/path/to/va-data/clones/seedvc")
    ap.add_argument("--n-enroll", type=int, default=8)
    args = ap.parse_args()

    actors = agency_actor_segments(min_segs=args.n_enroll + 4)  # need [:8] enroll + [8:12] held-out
    clones = sorted(Path(args.clone_dir).glob("clone_*__src*.wav"))
    clone_targets = []
    for c in clones:
        m = re.match(r"clone_(.+)__src\d+", c.stem)
        if m and m.group(1) in actors:
            clone_targets.append((m.group(1), c))
    tset = sorted({t for t, _ in clone_targets})
    print(f"clones {len(clone_targets)} over {len(tset)} targets (each with >= {args.n_enroll+4} real segs)")

    # ---- ECAPA raw cosine ----
    ext = build_extractor("ecapa", "cuda")
    cent = {}
    for t in tset:
        es = np.stack([ecapa_emb(ext, p) for _, _, p in actors[t][:args.n_enroll]])
        c = es.mean(0); cent[t] = c / (np.linalg.norm(c) + 1e-9)
    clone_tgt = [float(ecapa_emb(ext, c) @ cent[t]) for t, c in clone_targets]
    real_same = []
    for t in tset:
        for _, _, p in actors[t][args.n_enroll:args.n_enroll + 4]:
            real_same.append(float(ecapa_emb(ext, p) @ cent[t]))
    clone_tgt, real_same = np.array(clone_tgt), np.array(real_same)
    print(f"\n[ECAPA raw cosine]")
    print(f"  clone -> target centroid : {clone_tgt.mean():.3f} ± {clone_tgt.std():.3f}  "
          f"(p10 {np.percentile(clone_tgt,10):.3f} p90 {np.percentile(clone_tgt,90):.3f})")
    print(f"  real  -> own centroid    : {real_same.mean():.3f} ± {real_same.std():.3f}  (same-speaker baseline)")

    del ext
    torch.cuda.empty_cache()

    # ---- Resemblyzer ----
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except Exception as exc:  # noqa: BLE001
        print(f"\n[Resemblyzer] unavailable: {exc}")
        return
    venc = VoiceEncoder()

    def r_emb(p):
        return venc.embed_utterance(preprocess_wav(Path(p)))  # unit-norm

    rcent = {}
    for t in tset:
        es = np.stack([r_emb(p) for _, _, p in actors[t][:args.n_enroll]])
        c = es.mean(0); rcent[t] = c / (np.linalg.norm(c) + 1e-9)
    r_clone = [float(r_emb(c) @ rcent[t]) for t, c in clone_targets]
    r_real = []
    for t in tset:
        for _, _, p in actors[t][args.n_enroll:args.n_enroll + 4]:
            r_real.append(float(r_emb(p) @ rcent[t]))
    r_clone, r_real = np.array(r_clone), np.array(r_real)
    print(f"\n[Resemblyzer]")
    print(f"  clone -> target centroid : {r_clone.mean():.3f} ± {r_clone.std():.3f}  "
          f"(p10 {np.percentile(r_clone,10):.3f} p90 {np.percentile(r_clone,90):.3f})")
    print(f"  real  -> own centroid    : {r_real.mean():.3f} ± {r_real.std():.3f}  (same-speaker baseline)")

    import json
    out = {
        "clone_dir": args.clone_dir, "n_clones": len(clone_targets), "n_targets": len(tset),
        "ecapa_clone_to_target_mean": float(clone_tgt.mean()), "ecapa_clone_to_target_std": float(clone_tgt.std()),
        "ecapa_real_same_speaker_mean": float(real_same.mean()),
        "resemblyzer_clone_to_target_mean": float(r_clone.mean()), "resemblyzer_clone_to_target_std": float(r_clone.std()),
        "resemblyzer_clone_p10": float(np.percentile(r_clone, 10)), "resemblyzer_clone_p90": float(np.percentile(r_clone, 90)),
        "resemblyzer_real_same_speaker_mean": float(r_real.mean()),
    }
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/vc_fidelity.json", "w"), indent=2)
    print("\nwrote output/analysis/vc_fidelity.json")


if __name__ == "__main__":
    main()

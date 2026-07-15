"""Sanity/quality check for the Seed-VC replacement vs the old kNN-VC clones.
For each clone, cosine similarity of its embedding to (a) the TARGET actor's real
centroid (should be HIGH if the VC captured the target's timbre) and (b) the SOURCE
JVS speaker's centroid (should be LOW). A better VC pulls clones toward the target and
away from the source. Reports the distributions for both clone sets.

Usage: python -m src.vc_quality_check --models ecapa animeva
"""
from __future__ import annotations
import argparse, re
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, soundfile as sf
from src.clone_probe import va_actor_segments
from src.embeddings import build_extractor

SETS = {"seedvc": "/path/to/va-data/clones/seedvc",
        "knnvc": "/path/to/va-data/clones/knnvc"}
JVS_SEG = Path("/path/to/va-data/processed/jvs")


def emb(ext, p):
    w, _ = sf.read(str(p), dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
    return np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    args = ap.parse_args()
    actors = va_actor_segments()
    # source JVS speaker centroid: the clones all use jvs001 utterances as source
    src_files = sorted(JVS_SEG.glob("*.wav"))[:12]
    for model in args.models:
        ext = build_extractor(model, "cuda")
        tgt_centroid = {}
        src_c = np.mean([emb(ext, p) for p in src_files], 0)
        src_c /= np.linalg.norm(src_c) + 1e-9
        print(f"\n#### {model} ####")
        for name, d in SETS.items():
            if not Path(d).exists():
                continue
            to_tgt, to_src = [], []
            for f in sorted(Path(d).glob("clone_*__src*.wav")):
                m = re.match(r"clone_(.+)__src\d+", f.stem)
                T = m.group(1)
                if T not in actors:
                    continue
                if T not in tgt_centroid:
                    c = np.mean([emb(ext, p) for p in actors[T][:8]], 0)
                    tgt_centroid[T] = c / (np.linalg.norm(c) + 1e-9)
                v = emb(ext, f); v /= np.linalg.norm(v) + 1e-9
                to_tgt.append(float(v @ tgt_centroid[T])); to_src.append(float(v @ src_c))
            if to_tgt:
                tt, ts = np.array(to_tgt), np.array(to_src)
                print(f"  {name:8s} n={len(tt):3d}  cos→TARGET {tt.mean():.3f}±{tt.std():.3f}  "
                      f"cos→SOURCE {ts.mean():.3f}  target-minus-source {tt.mean()-ts.mean():+.3f}")


if __name__ == "__main__":
    main()

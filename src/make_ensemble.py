"""Build ensemble embeddings via score-level (mean-cosine) fusion.

Concatenating each model's L2-normalized embedding and re-normalizing yields a
space whose cosine similarity equals the *mean* of the per-model cosine
similarities -- i.e. standard score-level fusion. We then run the ordinary
margin/geometry/hubness analysis on the fused space.

Produces output/embeddings/<name>.npz for each requested ensemble, aligned on
the segments common to all member models.

Usage:
    python -m src.make_ensemble                      # default: all6 and sv4
    python -m src.make_ensemble --name ens_sv --models xvector ecapa campp redimnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EMB_DIR = Path("output/embeddings")

# Predefined ensembles.
ENSEMBLES = {
    "ens_all6": ["xvector", "ecapa", "wavlm", "campp", "redimnet", "jhubert"],
    "ens_sv4": ["xvector", "ecapa", "campp", "redimnet"],  # speaker-verification models only
}


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build(name: str, members: list[str]) -> None:
    loaded = {}
    for m in members:
        p = EMB_DIR / f"{m}.npz"
        if not p.exists():
            raise SystemExit(f"missing {p}; extract model '{m}' first")
        loaded[m] = np.load(p, allow_pickle=True)

    # Align on the intersection of segment_ids, preserving the first member's order.
    id_sets = [set(loaded[m]["segment_id"].tolist()) for m in members]
    common = set.intersection(*id_sets)
    base = loaded[members[0]]
    order = [s for s in base["segment_id"].tolist() if s in common]
    pos = {s: i for i, s in enumerate(order)}

    blocks = []
    for m in members:
        d = loaded[m]
        idx = np.full(len(order), -1, dtype=np.int64)
        for i, s in enumerate(d["segment_id"].tolist()):
            if s in pos:
                idx[pos[s]] = i
        emb = l2norm(d["emb"].astype(np.float32))[idx]   # reorder to common order
        blocks.append(emb)
    fused = np.concatenate(blocks, axis=1)  # cosine on this == mean per-model cosine

    # Metadata from the first member, reordered.
    d0 = base
    idx0 = np.array([np.where(d0["segment_id"] == s)[0][0] for s in order])
    np.savez(
        EMB_DIR / f"{name}.npz",
        emb=fused,
        segment_id=np.array(order),
        speaker_id=d0["speaker_id"][idx0],
        style_label=d0["style_label"][idx0],
        recording_source=(d0["recording_source"][idx0] if "recording_source" in d0
                          else np.array(["unknown"] * len(order))),
    )
    meta = {"model": name, "ensemble_of": members, "num_segments": len(order),
            "num_speakers": int(len(set(d0["speaker_id"][idx0].tolist()))),
            "fused_dim": int(fused.shape[1]), "fusion": "mean-cosine (concat L2-normalized)"}
    with (EMB_DIR / f"{name}.meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"  {name}: {fused.shape} from {members} ({len(order)} common segments)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ensemble (mean-cosine fusion) embeddings.")
    parser.add_argument("--name", default=None, help="Single ensemble name (with --models).")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    if args.name and args.models:
        build(args.name, args.models)
    else:
        for name, members in ENSEMBLES.items():
            build(name, members)


if __name__ == "__main__":
    main()

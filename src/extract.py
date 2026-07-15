"""Phase 1: extract speaker embeddings for every segment with every model.

Reads data/processed/segments.jsonl, computes an embedding per segment for each
requested model, and writes one .npz per model under output/embeddings/:

    output/embeddings/<model>.npz
        emb         float32 (N, D)   -- raw embeddings (not normalized)
        segment_id  str    (N,)
        speaker_id  str    (N,)
        style_label str    (N,)
    output/embeddings/<model>.meta.json   -- model spec + counts + timestamp

Usage:
    python -m src.extract [--models ecapa wavlm ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src.embeddings import MODEL_SPECS, build_extractor
# Collection modules (src/scrape/) are not part of this release; local shim:
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-06-23T05:30:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

SEGMENTS_JSONL = Path("data/processed/segments.jsonl")
OUT_DIR = Path("output/embeddings")


def load_segments(limit: int | None) -> list[dict]:
    if not SEGMENTS_JSONL.exists():
        raise SystemExit(f"{SEGMENTS_JSONL} not found. Run Phase 0c (src.build_dataset) first.")
    with SEGMENTS_JSONL.open(encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if "segment_id" in r]
    return rows[:limit] if limit else rows


def read_wav(path: str) -> torch.Tensor:
    """Read a preprocessed 16 kHz mono WAV as a 1-D float32 tensor."""
    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav)


def extract_for_model(name: str, segments: list[dict], device: str) -> None:
    ext = build_extractor(name, device)
    embs, seg_ids, spk_ids, styles, sources = [], [], [], [], []
    for r in tqdm(segments, desc=f"extract:{name}"):
        try:
            wav = read_wav(r["segment_path"])
            v = ext.embed(wav)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {r['segment_id']}: {exc}")
            continue
        embs.append(np.asarray(v, dtype=np.float32).reshape(-1))
        seg_ids.append(r["segment_id"])
        spk_ids.append(r["speaker_id"])
        styles.append(r.get("style_label", ""))
        sources.append(r.get("recording_source", r.get("agency", "unknown")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_arr = np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)
    np.savez(
        OUT_DIR / f"{name}.npz",
        emb=emb_arr,
        segment_id=np.array(seg_ids),
        speaker_id=np.array(spk_ids),
        style_label=np.array(styles),
        recording_source=np.array(sources),
    )
    spec = MODEL_SPECS[name]
    meta = {
        "model": name,
        "paradigm": spec.paradigm,
        "train_lang": spec.train_lang,
        "generation": spec.generation,
        "embedding_dim": int(emb_arr.shape[1]) if emb_arr.size else 0,
        "num_segments": int(emb_arr.shape[0]),
        "num_speakers": int(len(set(spk_ids))),
        "extracted_at_utc": utc_now_iso(),
    }
    with (OUT_DIR / f"{name}.meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"  -> {name}: {emb_arr.shape} for {meta['num_speakers']} speakers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speaker embeddings for all models.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()),
                        choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--limit", type=int, default=None, help="Only first N segments (debug).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    segments = load_segments(args.limit)
    print(f"Loaded {len(segments)} segments; models: {args.models}; device: {args.device}")
    for name in args.models:
        extract_for_model(name, segments, args.device)


if __name__ == "__main__":
    main()

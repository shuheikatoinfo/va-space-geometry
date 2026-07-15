"""Phase 0c: audio preprocessing.

Reads the collected agency voice samples (data/agency_voices/samples.jsonl),
and for each source file produces uniform analysis segments:

  - 16 kHz, mono, 16-bit PCM WAV
  - energy-based silence removal (a "simple VAD")
  - uniform fixed-length segments (default 4 s, in the 3-5 s band); the trailing
    remainder shorter than the minimum is discarded

Each output segment keeps a back-pointer to its source file's provenance
(source URL + SHA-256), so every analysis data point is traceable to when and
where it was downloaded.

Outputs:
  data/processed/<speaker_id>/<sample_uid>__seg<k>.wav
  data/processed/segments.jsonl   (one record per segment)
  data/processed/manifest.json

Usage:
    python -m src.build_dataset [--seg-sec 4.0] [--min-sec 3.0] [--top-db 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Collection modules (src/scrape/) are not part of this release; local shim:
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-06-23T05:30:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("librosa is required for preprocessing: pip install librosa") from exc

TARGET_SR = 16000
# Phase 0b writes one samples.jsonl per source (agency + freelance); read all.
SAMPLE_SOURCES = [
    Path("data/agency_voices/samples.jsonl"),
    Path("data/freelance/samples.jsonl"),
]
OUT_DIR = Path("data/processed")
STAGE_VERSION = "1.0"


def load_mono_16k(path: Path) -> np.ndarray:
    """Load any audio file as float32 mono at 16 kHz (ffmpeg via librosa)."""
    wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return wav.astype(np.float32)


def remove_silence(wav: np.ndarray, top_db: float) -> np.ndarray:
    """Concatenate non-silent regions (simple energy-based VAD)."""
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    return np.concatenate([wav[s:e] for s, e in intervals])


def segment(wav: np.ndarray, seg_sec: float, min_sec: float) -> list[np.ndarray]:
    """Cut into consecutive fixed-length windows; drop trailing remainder < min."""
    seg_len = int(seg_sec * TARGET_SR)
    min_len = int(min_sec * TARGET_SR)
    segments = []
    for start in range(0, len(wav), seg_len):
        chunk = wav[start:start + seg_len]
        if len(chunk) >= min_len:
            segments.append(chunk)
    return segments


def iter_samples() -> list[dict]:
    samples: list[dict] = []
    for src in SAMPLE_SOURCES:
        if src.exists():
            with src.open(encoding="utf-8") as fh:
                samples.extend(json.loads(line) for line in fh if line.strip())
    if not samples:
        raise SystemExit(
            "No samples found. Run Phase 0b (agency_voices / freelance_voices) first."
        )
    return samples


def recording_source_of(sample: dict) -> str:
    """Recording-environment label for the channel-confound analysis.

    Freelance records carry an explicit recording_source (platform/domain);
    agency records use the agency as a proxy for a shared studio/recording chain.
    """
    return sample.get("recording_source") or sample.get("agency") or "unknown"


_PARAMS = {"seg_sec": 4.0, "min_sec": 3.0, "top_db": 30.0}


def process_one(s: dict) -> dict:
    """Worker: load -> VAD -> segment -> write WAVs; return records for one sample.

    Returns {"records": [...], "error": str|None}. Designed for a process pool.
    """
    src_path = Path(s["local_path"])
    sample_uid = s["sample_uid"]
    speaker_id = s["speaker_id"]
    try:
        wav = load_mono_16k(src_path)
        voiced = remove_silence(wav, _PARAMS["top_db"])
        chunks = segment(voiced, _PARAMS["seg_sec"], _PARAMS["min_sec"])
    except Exception as exc:  # noqa: BLE001
        return {"records": [], "error": f"{sample_uid}: {exc}"}

    out_dir = OUT_DIR / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for k, chunk in enumerate(chunks):
        out_path = out_dir / f"{sample_uid}__seg{k:02d}.wav"
        sf.write(str(out_path), chunk, TARGET_SR, subtype="PCM_16")
        records.append({
            "segment_id": f"{sample_uid}__seg{k:02d}",
            "speaker_id": speaker_id,
            "speaker_name": s.get("speaker_name", ""),
            "agency": s.get("agency", ""),
            "recording_source": recording_source_of(s),
            "style_label": s.get("style_label", ""),
            "segment_path": str(out_path),
            "duration_sec": round(len(chunk) / TARGET_SR, 3),
            "source_url": s.get("source_url", ""),
            "source_sha256": s.get("content_sha256", ""),
            "source_fetched_at_utc": s.get("fetched_at_utc", ""),
        })
    return {"records": records, "error": None}


def main() -> None:
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Preprocess agency voice samples into segments.")
    parser.add_argument("--seg-sec", type=float, default=4.0, help="Segment length (seconds).")
    parser.add_argument("--min-sec", type=float, default=3.0, help="Minimum kept segment length.")
    parser.add_argument("--top-db", type=float, default=30.0, help="VAD threshold below ref (dB).")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (debug).")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = parser.parse_args()
    _PARAMS.update(seg_sec=args.seg_sec, min_sec=args.min_sec, top_db=args.top_db)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = iter_samples()
    if args.limit:
        samples = samples[: args.limit]

    started = utc_now_iso()
    seg_records: list[dict] = []
    n_fail = 0
    speakers_with_audio: set[str] = set()

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_one, samples, chunksize=8),
                            total=len(samples), desc="preprocess"):
                if res["error"]:
                    n_fail += 1
                    seg_records.append({"error": res["error"]})
                for r in res["records"]:
                    seg_records.append(r)
                    speakers_with_audio.add(r["speaker_id"])
    else:
        for s in tqdm(samples, desc="preprocess"):
            res = process_one(s)
            if res["error"]:
                n_fail += 1
                seg_records.append({"error": res["error"]})
            for r in res["records"]:
                seg_records.append(r)
                speakers_with_audio.add(r["speaker_id"])

    finished = utc_now_iso()
    seg_path = OUT_DIR / "segments.jsonl"
    with seg_path.open("w", encoding="utf-8") as fh:
        for r in seg_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_segments = sum(1 for r in seg_records if "segment_id" in r)
    total_dur = sum(r.get("duration_sec", 0) for r in seg_records if "segment_id" in r)
    manifest = {
        "stage": "0c_preprocess",
        "stage_version": STAGE_VERSION,
        "started_utc": started,
        "finished_utc": finished,
        "params": {"seg_sec": args.seg_sec, "min_sec": args.min_sec, "top_db": args.top_db,
                   "target_sr": TARGET_SR},
        "num_source_samples": len(samples),
        "num_segments": n_segments,
        "num_speakers_with_segments": len(speakers_with_audio),
        "total_segment_duration_sec": round(total_dur, 1),
        "num_failures": n_fail,
        "outputs": {"segments_jsonl": str(seg_path), "processed_dir": str(OUT_DIR)},
    }
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"Segments: {n_segments} from {len(speakers_with_audio)} speakers, "
          f"{total_dur/60:.1f} min total, {n_fail} failures.")


if __name__ == "__main__":
    main()

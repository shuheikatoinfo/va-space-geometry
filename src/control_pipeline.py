"""Build + extract embeddings for a CONTROL speaker population (JVS / Common Voice
Japanese), to test whether the voice-actor domain is unusually high-density.

For each corpus we take K speakers x ~M utterances, preprocess identically to the
voice-actor set (16 kHz mono, VAD, 3-5 s segments), and extract the same speaker
embeddings. Output: output/embeddings_control/<corpus>_<model>.npz (same schema as
the VA embeddings) so the matched margin/hubness comparison reuses the analysis.

Usage:
    python -m src.control_pipeline --corpus jvs --models ecapa jxvector animeva
    python -m src.control_pipeline --corpus cv  --models ecapa jxvector animeva --max-speakers 800
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.build_dataset import load_mono_16k, remove_silence, segment, TARGET_SR
from src.embeddings import build_extractor

CORPORA_ROOT = Path("/path/to/va-data")
SEG_ROOT = CORPORA_ROOT / "processed"
OUT_DIR = Path("output/embeddings_control")
JVS_ROOT = CORPORA_ROOT / "jvs" / "jvs_ver1"


def enumerate_jvs(utts_per_speaker: int, varied: bool = False) -> list[tuple[str, str]]:
    """Return (speaker_id, wav_path) for JVS utterances.

    varied=False: parallel100 (all speakers read the SAME sentences, neutral) ->
        minimal intra-speaker variation (easy control).
    varied=True: mix nonpara30 (different sentences) + falsetto10 + whisper10 ->
        intra-speaker STYLE variation, a fairer match to voice-actor style range.
    """
    pairs = []
    for spk_dir in sorted(JVS_ROOT.glob("jvs[0-9][0-9][0-9]")):
        sid = f"jvs_{spk_dir.name}"
        if not varied:
            wavs = sorted((spk_dir / "parallel100" / "wav24kHz16bit").glob("*.wav"))[:utts_per_speaker]
        else:
            wavs = []
            for sub, n in [("nonpara30", 8), ("falset10", 2), ("whisper10", 2)]:
                wavs += sorted((spk_dir / sub / "wav24kHz16bit").glob("*.wav"))[:n]
        for w in wavs:
            pairs.append((sid, str(w)))
    return pairs


def build_cv_segments(utts_per_speaker: int, max_speakers: int, seg_sec=4.0, min_sec=3.0, top_db=30.0):
    """Stream Common Voice JA, collect speakers with >= utts, write 16 kHz segments.

    Streaming avoids materializing the whole split; we decode each clip's audio
    array inline and write it, grouping by client_id until we have enough speakers.
    """
    import soundfile as sf
    from datasets import load_dataset

    seg_dir = SEG_ROOT / "cv"
    seg_dir.mkdir(parents=True, exist_ok=True)
    from datasets import Audio
    buf = defaultdict(list)   # client_id -> list of (uid, np.array)
    done = set()
    records = []
    seen = 0
    for split in ["other", "train", "validation", "test"]:
        if len(done) >= max_speakers:
            break
        ds = load_dataset("fsicoli/common_voice_17_0", "ja", split=split,
                          streaming=True, trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        for row in ds:
            seen += 1
            cid = row["client_id"]
            if cid in done:
                continue
            buf[cid].append((row["path"], row["audio"]["array"].astype(np.float32)))
            if len(buf[cid]) >= utts_per_speaker:
                spk = f"cv_{cid[:16]}"
                for uid, arr in buf[cid]:
                    voiced = remove_silence(arr, top_db)
                    for k, c in enumerate(segment(voiced, seg_sec, min_sec)):
                        stem = "".join(ch if ch.isalnum() else "_" for ch in Path(uid).stem)[:30]
                        outp = seg_dir / f"{spk}__{stem}__seg{k:02d}.wav"
                        sf.write(str(outp), c, TARGET_SR, subtype="PCM_16")
                        records.append({"segment_id": f"{spk}__{stem}__seg{k:02d}", "speaker_id": spk,
                                        "recording_source": "control:cv", "segment_path": str(outp)})
                done.add(cid)
                buf.pop(cid, None)
                if len(done) >= max_speakers:
                    break
            if seen % 20000 == 0:
                print(f"  [{split}] scanned {seen} clips, {len(done)} speakers")
    print(f"  CV: scanned {seen} clips -> {len(done)} speakers")
    return records


def build_segments(pairs, corpus: str, seg_sec=4.0, min_sec=3.0, top_db=30.0) -> list[dict]:
    seg_dir = SEG_ROOT / corpus
    seg_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spk, path in tqdm(pairs, desc=f"prep:{corpus}"):
        try:
            wav = load_mono_16k(Path(path))
            voiced = remove_silence(wav, top_db)
            chunks = segment(voiced, seg_sec, min_sec)
        except Exception:
            continue
        uid = Path(path).stem
        for k, c in enumerate(chunks):
            import soundfile as sf
            outp = seg_dir / f"{spk}__{uid}__seg{k:02d}.wav"
            sf.write(str(outp), c, TARGET_SR, subtype="PCM_16")
            records.append({"segment_id": f"{spk}__{uid}__seg{k:02d}", "speaker_id": spk,
                            "recording_source": f"control:{corpus}", "segment_path": str(outp)})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["jvs", "jvsv", "cv"], required=True)
    ap.add_argument("--models", nargs="+", default=["ecapa", "jxvector", "animeva"])
    ap.add_argument("--utts", type=int, default=12)
    ap.add_argument("--max-speakers", type=int, default=800)
    args = ap.parse_args()

    seg_json = SEG_ROOT / f"{args.corpus}_segments.jsonl"
    if seg_json.exists():
        records = [json.loads(l) for l in seg_json.open()]
        print(f"reusing {len(records)} cached segments")
    else:
        if args.corpus == "cv":
            records = build_cv_segments(args.utts, args.max_speakers)
        else:
            pairs = enumerate_jvs(args.utts, varied=(args.corpus == "jvsv"))
            print(f"{args.corpus}: {len(pairs)} utterances, {len(set(p[0] for p in pairs))} speakers")
            records = build_segments(pairs, args.corpus)
        seg_json.parent.mkdir(parents=True, exist_ok=True)
        with seg_json.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} segments, {len(set(r['speaker_id'] for r in records))} speakers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    for name in args.models:
        ext = build_extractor(name, "cuda" if torch.cuda.is_available() else "cpu")
        embs, sids, srcs, segids = [], [], [], []
        for r in tqdm(records, desc=f"extract:{args.corpus}:{name}"):
            try:
                w, _ = sf.read(r["segment_path"], dtype="float32")
                if w.ndim > 1:
                    w = w.mean(1)
                v = ext.embed(torch.from_numpy(w))
            except Exception:
                continue
            embs.append(np.asarray(v, np.float32).reshape(-1)); sids.append(r["speaker_id"])
            srcs.append(r["recording_source"]); segids.append(r["segment_id"])
        np.savez(OUT_DIR / f"{args.corpus}_{name}.npz", emb=np.vstack(embs),
                 speaker_id=np.array(sids), recording_source=np.array(srcs), segment_id=np.array(segids))
        print(f"  -> {args.corpus}_{name}: {np.vstack(embs).shape}")


if __name__ == "__main__":
    main()

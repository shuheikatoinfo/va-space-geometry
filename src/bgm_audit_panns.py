"""BGM audit v2: AudioSet-trained tagger (PANNs CNN14) music probability.

For each agency source file, run PANNs audio tagging on up to 60 s (32 kHz)
and record the max clipwise probability over music-related AudioSet classes.
Clone files (known music-free) are the negative control; JVS reads (also
music-free studio speech) could be added similarly if needed.

Output: output/analysis/bgm_audit_panns.json
"""
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from panns_inference import AudioTagging, labels

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data/agency_voices/samples.jsonl"
CLONE_DIR = Path("/path/to/va-data/clones")
OUT = ROOT / "output/analysis/bgm_audit_panns.json"

SR = 32000
MAX_SEC = 60.0
BATCH = 8
MUSIC_LABELS = [i for i, l in enumerate(labels)
                if l in ("Music", "Musical instrument", "Background music",
                         "Theme music", "Soundtrack music", "Jingle (music)")]
THRESH = [0.3, 0.5, 0.7]


def load(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_SEC)
        if len(y) < SR:
            return None
        return y
    except Exception:
        return None


def score_files(at, files, meta=None):
    rows = []
    buf, keep = [], []
    maxlen = int(SR * MAX_SEC)

    def flush():
        if not buf:
            return
        n = max(len(b) for b in buf)
        batch = np.zeros((len(buf), n), dtype=np.float32)
        for i, b in enumerate(buf):
            batch[i, :len(b)] = b
        with torch.no_grad():
            clipwise, _ = at.inference(batch)
        for i, p in enumerate(keep):
            probs = clipwise[i]
            rows.append({
                "path": str(p),
                "music_prob": float(max(probs[j] for j in MUSIC_LABELS)),
                "speech_prob": float(probs[labels.index("Speech")]),
                "agency": (meta or {}).get(str(p), ""),
            })
        buf.clear()
        keep.clear()

    for k, p in enumerate(files):
        y = load(p)
        if y is None:
            continue
        buf.append(y[:maxlen])
        keep.append(p)
        if len(buf) >= BATCH:
            flush()
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)
    flush()
    return rows


def main():
    files, meta = [], {}
    with open(SAMPLES) as f:
        for line in f:
            r = json.loads(line)
            p = r.get("local_path")
            if p and (ROOT / p).exists():
                files.append(ROOT / p)
                meta[str(ROOT / p)] = r.get("agency_slug", "?")
    clones = sorted(CLONE_DIR.rglob("*.wav")) if CLONE_DIR.exists() else []
    random.seed(0)
    clone_sample = random.sample(clones, min(300, len(clones)))
    print(f"agency files: {len(files)}, clone controls: {len(clone_sample)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    at = AudioTagging(checkpoint_path=None, device=device)

    agency_rows = score_files(at, files, meta)
    clone_rows = score_files(at, clone_sample)

    summary = {}
    for th in THRESH:
        a = [r["music_prob"] > th for r in agency_rows]
        c = [r["music_prob"] > th for r in clone_rows]
        summary[f"music_prob>{th}"] = {
            "agency_bgm_rate": round(float(np.mean(a)), 4), "agency_n": len(a),
            "clone_control_rate": round(float(np.mean(c)), 4), "clone_n": len(c),
        }
    per_agency = {}
    for r in agency_rows:
        d = per_agency.setdefault(r["agency"], [0, 0])
        d[0] += int(r["music_prob"] > 0.5)
        d[1] += 1
    per_agency = {k: {"rate": round(v[0] / v[1], 3), "n": v[1]}
                  for k, v in sorted(per_agency.items(),
                                     key=lambda kv: -kv[1][0] / max(1, kv[1][1]))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "per_agency": per_agency,
                   "agency_files": agency_rows, "clone_controls": clone_rows}, f)
    print(json.dumps({"summary": summary,
                      "top_agencies": dict(list(per_agency.items())[:10])},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

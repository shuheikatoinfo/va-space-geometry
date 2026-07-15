"""BGM (music-bed) audit for agency demo-reel source files.

Heuristic: a music bed keeps playing through speech pauses, so files with BGM
have (a) a high residual energy floor in low-RMS "pause" frames relative to the
speech peak, and (b) tonal (low spectral-flatness) content in those frames.
Clone files (known music-free) serve as a negative control.

Outputs output/analysis/bgm_audit.json with per-file scores and flag rates.
"""
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data/agency_voices/samples.jsonl"
CLONE_DIR = Path("/path/to/va-data/clones")
OUT = ROOT / "output/analysis/bgm_audit.json"

MAX_SEC = 90.0
SR = 22050
PAUSE_DB_BELOW_PEAK = 25.0  # frames this far below speech peak count as pauses
# flag thresholds (primary + sensitivity)
FLOOR_DB_THRESH = [-45.0, -40.0, -50.0]
FLATNESS_THRESH = 0.2


def analyze(path):
    import librosa
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_SEC)
        if len(y) < SR:
            return None
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        peak = np.percentile(rms, 95)
        if peak <= 0:
            return None
        pause = rms < peak * 10 ** (-PAUSE_DB_BELOW_PEAK / 20)
        if pause.sum() < 20:  # <~0.5 s of pauses: fall back to quietest decile
            pause = rms < np.percentile(rms, 10)
        floor_db = float(20 * np.log10(max(np.median(rms[pause]), 1e-10) / peak))
        flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=512)[0]
        n = min(len(flat), len(pause))
        flat_med = float(np.median(flat[:n][pause[:n]]))
        return {"path": str(path), "pause_floor_db": floor_db,
                "pause_flatness": flat_med, "pause_frac": float(pause.mean())}
    except Exception as e:
        return {"path": str(path), "error": str(e)}


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
    clone_sample = random.sample(clones, min(200, len(clones)))
    print(f"agency files: {len(files)}, clone controls: {len(clone_sample)}")

    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        agency_res = [r for r in pool.imap_unordered(analyze, files, 16) if r]
        clone_res = [r for r in pool.imap_unordered(analyze, clone_sample, 16) if r]

    def flag_rate(rows, floor_th):
        ok = [r for r in rows if "error" not in r]
        flags = [r["pause_floor_db"] > floor_th and r["pause_flatness"] < FLATNESS_THRESH
                 for r in ok]
        return sum(flags) / max(1, len(flags)), len(ok)

    summary = {}
    for th in FLOOR_DB_THRESH:
        a_rate, a_n = flag_rate(agency_res, th)
        c_rate, c_n = flag_rate(clone_res, th)
        summary[f"floor>{th}dB & flatness<{FLATNESS_THRESH}"] = {
            "agency_bgm_rate": round(a_rate, 4), "agency_n": a_n,
            "clone_control_rate": round(c_rate, 4), "clone_n": c_n,
        }
    # per-agency rates at primary threshold
    per_agency = {}
    for r in agency_res:
        if "error" in r:
            continue
        slug = meta.get(r["path"], "?")
        flag = r["pause_floor_db"] > FLOOR_DB_THRESH[0] and r["pause_flatness"] < FLATNESS_THRESH
        d = per_agency.setdefault(slug, [0, 0])
        d[0] += int(flag)
        d[1] += 1
    per_agency = {k: {"rate": round(v[0] / v[1], 3), "n": v[1]}
                  for k, v in sorted(per_agency.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1]))}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "per_agency": per_agency,
                   "agency_files": agency_res, "clone_controls": clone_res}, f)
    print(json.dumps({"summary": summary,
                      "top_agencies": dict(list(per_agency.items())[:10])},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

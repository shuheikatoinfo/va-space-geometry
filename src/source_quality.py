"""Acoustic-quality evidence per recording source (is agency audio really studio-
grade?). On the ORIGINAL downloaded files we estimate: sample rate, effective
bandwidth (99%-energy spectral rolloff), and a noise-floor SNR proxy
(median frame energy vs 10th-percentile / silence energy). Higher SNR + fuller
bandwidth + no reverb ≈ professional recording.

Usage: python -m src.source_quality
"""
from __future__ import annotations

import glob, json, random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

random.seed(0)
# agency sources are de-identified as A/B/C (real agency names withheld);
# substitute your own source globs to reproduce.
SOURCES = {
    "agency A (mp3)": "data/registry/agency_a/**/*.mp3",
    "agency B (mp3)": "data/registry/agency_b/**/*.mp3",
    "agency C (mp3)": "data/registry/agency_c/**/*.mp3",
    "freelance youtube": "data/registry/freelance/**/*.webm",
    "freelance direct": "data/registry/freelance/**/*.mp3",
    "JVS (studio ref)": "/path/to/va-data/jvs/jvs_ver1/*/parallel100/wav24kHz16bit/*.wav",
    "CommonVoice (crowd)": "/path/to/va-data/processed/cv/*.wav",
}


def metrics(path, n_fft=1024):
  try:
    try:
        w, sr = sf.read(path, dtype="float32")
    except Exception:
        w, sr = librosa.load(path, sr=None, mono=True)
    if getattr(w, "ndim", 1) > 1:
        w = w.mean(1)
    w = np.asarray(w, dtype=np.float32)
    if w.size < max(sr // 2, 800) or not np.isfinite(w).all():
        return None
    # effective bandwidth: highest freq holding 99% of cumulative energy
    S = np.abs(librosa.stft(w, n_fft=n_fft)) ** 2
    psd = S.mean(1); cum = np.cumsum(psd) / psd.sum()
    roll = np.searchsorted(cum, 0.99) / len(cum) * (sr / 2)
    # SNR proxy: frame RMS; signal=median of voiced frames, noise=10th pct
    fr = librosa.util.frame(w, frame_length=400, hop_length=200)
    e = 20 * np.log10(np.sqrt((fr ** 2).mean(0)) + 1e-9)
    snr = float(np.percentile(e, 75) - np.percentile(e, 10))
    return sr, float(roll), snr
  except Exception:
    return None


def main():
    rows = {}
    for name, pat in SOURCES.items():
        files = glob.glob(pat, recursive=True)
        random.shuffle(files); files = files[:60]
        srs, rolls, snrs = [], [], []
        for f in files:
            m = metrics(f)
            if m:
                srs.append(m[0]); rolls.append(m[1]); snrs.append(m[2])
        if snrs:
            rows[name] = {"n": len(snrs), "median_sr": int(np.median(srs)),
                          "median_bandwidth_hz": int(np.median(rolls)),
                          "median_snr_proxy_db": round(float(np.median(snrs)), 1)}
    print(f"{'source':24s} {'n':>3s} {'sr':>6s} {'bw(Hz)':>7s} {'SNR~dB':>7s}")
    for k, v in rows.items():
        print(f"{k:24s} {v['n']:3d} {v['median_sr']:6d} {v['median_bandwidth_hz']:7d} {v['median_snr_proxy_db']:7.1f}")
    json.dump(rows, open("output/analysis/source_quality.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

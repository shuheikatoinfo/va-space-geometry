"""TASK 1 (§4.3 + §7 "mastering homogeneity" caveat): loudness / spectral-tilt
normalized re-embedding of the codec-matched agency subset.

The paper worries professional demo reels share a broadcast mastering aesthetic
(loudness / EQ / limiting) that could pull clean pro voices together independent of
vocal-tract similarity, and defers "re-running with normalized loudness / spectral
tilt" to future work (§4.3, §7). This does it.

Conditions (per segment, applied to the 16 kHz mono waveform BEFORE re-embedding):
  - full        : cached raw embeddings (no normalization) -- the baseline, reused
                  from output/embeddings/ (exact same agency subset & alignment).
  - loudnorm    : (a) ITU-R BS.1770 loudness normalization to TARGET_LUFS via
                  pyloudnorm.
  - loud+tilt   : (a) loudness norm, then (b) spectral-tilt flattening: estimate the
                  long-term spectral slope (a linear fit, in dB, of the long-term
                  average log-magnitude spectrum vs frequency over [TILT_LO,TILT_HI])
                  and apply the inverse tilt as a zero-phase STFT-domain gain that
                  whitens that first-order (linear-in-frequency) slope to ~0 dB,
                  then re-normalize loudness to TARGET_LUFS (tilt EQ shifts LUFS).

We re-embed (normalization changes the waveform) with the repo's own extractors and
exact 16 kHz mono front-end, for ECAPA, animeva, and SV-4 (fusion of xvector, ECAPA,
CAM++, ReDimNet, per make_ensemble). Then recompute, per condition:
  - raw-cosine closed-set misID (analyze.py definition)
  - hubness skew (k-occurrence skewness)
  - same-agency nearest-neighbor enrichment (nearest impostor same recording_source
    fraction / chance), the §4.3 "nearest neighbor is same recording source" ratio.

If the misID floor and same-agency enrichment barely move, mastering aesthetics are
not the driver of the thin-margin geometry.

Usage:
    # sanity: one encoder, small speaker cap, dump 3 example WAVs
    python -m src.mastering_normalized_rerun --sanity
    # full run
    python -m src.mastering_normalized_rerun --models ecapa animeva ens_sv4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import stats

import pyloudnorm as pyln

from src.embeddings import build_extractor

SR = 16000
TARGET_LUFS = -23.0
TILT_LO, TILT_HI = 100.0, 7000.0   # band for the long-term slope fit
NFFT = 1024
HOP = 256
EMB = Path("output/embeddings")
AN = Path("output/analysis")
NORM_EMB = Path("output/embeddings_mastering")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SV-4 = score-level fusion of these members (make_ensemble.ENSEMBLES["ens_sv4"]).
SV4_MEMBERS = ["xvector", "ecapa", "campp", "redimnet"]

_METER = pyln.Meter(SR)


# ---------------------------------------------------------------- normalization
def loudness_norm(w: np.ndarray, target=TARGET_LUFS) -> np.ndarray:
    w = w.astype(np.float64)
    if np.max(np.abs(w)) < 1e-6:
        return w.astype(np.float32)
    try:
        loud = _METER.integrated_loudness(w)
    except Exception:
        return w.astype(np.float32)
    if not np.isfinite(loud) or loud < -70:  # silent / undefined
        return w.astype(np.float32)
    out = pyln.normalize.loudness(w, loud, target)
    peak = np.max(np.abs(out))
    if peak > 0.99:  # avoid clipping after gain
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def _stft(w):
    win = np.hanning(NFFT).astype(np.float32)
    n = 1 + (len(w) - NFFT) // HOP if len(w) >= NFFT else 0
    if n <= 0:
        return None
    frames = np.stack([w[i * HOP:i * HOP + NFFT] * win for i in range(n)])
    return np.fft.rfft(frames, axis=1)


def tilt_flatten(w: np.ndarray) -> np.ndarray:
    """Whiten the average first-order spectral tilt via a zero-phase STFT gain."""
    S = _stft(w.astype(np.float32))
    if S is None:
        return w
    mag = np.abs(S)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    lt = 20.0 * np.log10(np.mean(mag, axis=0) + 1e-8)  # long-term avg log-mag (dB)
    band = (freqs >= TILT_LO) & (freqs <= TILT_HI)
    if band.sum() < 4:
        return w
    slope, intercept = np.polyfit(freqs[band], lt[band], 1)  # dB per Hz
    fc = 0.5 * (TILT_LO + TILT_HI)
    gain_db = -slope * (freqs - fc)          # inverse tilt, unity at band center
    gain = (10.0 ** (gain_db / 20.0)).astype(np.float32)
    Sw = S * gain[None, :]
    # zero-phase overlap-add reconstruction
    win = np.hanning(NFFT).astype(np.float32)
    frames = np.fft.irfft(Sw, n=NFFT, axis=1).astype(np.float32)
    out = np.zeros(len(w), np.float32)
    wsum = np.zeros(len(w), np.float32)
    for i in range(frames.shape[0]):
        s = i * HOP
        out[s:s + NFFT] += frames[i] * win
        wsum[s:s + NFFT] += win * win
    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    return out


def apply_condition(w: np.ndarray, cond: str) -> np.ndarray:
    if cond == "loudnorm":
        return loudness_norm(w)
    if cond == "loud+tilt":
        return loudness_norm(tilt_flatten(loudness_norm(w)))
    raise ValueError(cond)


# ------------------------------------------------------------------- data load
def agency_segments():
    rows = [json.loads(l) for l in open("data/processed/segments.jsonl") if l.strip()]
    rows = [r for r in rows if "segment_id" in r
            and not str(r.get("recording_source", "")).startswith("freelance:")]
    return rows


def load_wav(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    assert sr == SR, f"{path}: {sr}Hz != {SR}"
    return np.asarray(w, np.float32)


# ---------------------------------------------------------------- metrics (raw cosine)
def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


def compute_metrics(emb, spk, src, k=10):
    """raw-cosine closed-set misID, hubness skew, same-source (agency) NN enrichment."""
    X = torch.from_numpy(l2(emb.astype(np.float32))).to(DEVICE)
    n = len(spk)
    _, codes = np.unique(spk, return_inverse=True)
    ct = torch.from_numpy(codes).to(DEVICE)
    neg = torch.tensor(-9.0, device=DEVICE)
    sb = np.full(n, -9.0, np.float32); db = np.full(n, -9.0, np.float32)
    db_idx = np.full(n, -1, np.int64)
    occ = np.zeros(n, np.int64)
    for st in range(0, n, 2048):
        e = min(st + 2048, n)
        sims = X[st:e] @ X.T
        sims[torch.arange(e - st, device=DEVICE), torch.arange(st, e, device=DEVICE)] = neg
        same = ct.unsqueeze(0) == ct[st:e].unsqueeze(1)
        sb[st:e] = torch.where(same, sims, neg).max(1).values.cpu().numpy()
        dv, di = torch.where(same, neg, sims).max(1)
        db[st:e] = dv.cpu().numpy(); db_idx[st:e] = di.cpu().numpy()
        idx = sims.topk(k, 1).indices.reshape(-1).cpu().numpy()
        np.add.at(occ, idx, 1)
    has = sb > -8.0
    misid = float(np.mean(sb[has] < db[has]))
    hub = float(stats.skew(occ.astype(float)))
    # same-agency (recording_source) nearest-impostor enrichment
    src = np.asarray(src)
    valid = db_idx >= 0
    comp = np.where(valid, src[db_idx.clip(min=0)], "")
    same_src = (comp == src) & valid
    _, codes2, cnt = np.unique(src, return_inverse=True, return_counts=True)
    chance = float(np.mean((cnt[codes2] - 1) / max(n - 1, 1)))
    nn_same = float(np.mean(same_src[valid]))
    return {"n": int(n), "misid": misid, "hubness_skew": hub,
            "nn_same_source_frac": nn_same, "chance_same_source": chance,
            "same_source_enrichment": (nn_same / chance) if chance else None}


# ------------------------------------------------------------------ embedding
def embed_condition(rows, cond, base_encoders):
    """Return {encoder: (N,D) array} for the given normalization condition."""
    exts = {e: build_extractor(e, DEVICE) for e in base_encoders}
    out = {e: [] for e in base_encoders}
    for i, r in enumerate(rows):
        w = load_wav(r["segment_path"])
        w = apply_condition(w, cond) if cond != "full" else w
        wt = torch.from_numpy(np.ascontiguousarray(w, np.float32))
        for e in base_encoders:
            v = np.asarray(exts[e].embed(wt), np.float32).reshape(-1)
            out[e].append(v)
        if (i + 1) % 2000 == 0:
            print(f"    [{cond}] {i+1}/{len(rows)}", flush=True)
    del exts
    torch.cuda.empty_cache()
    return {e: np.vstack(v) for e, v in out.items()}


def fuse_sv4(member_embs):
    """Score-level fusion: concat L2-normalized members, renormalize (make_ensemble)."""
    blocks = [l2(member_embs[m].astype(np.float32)) for m in SV4_MEMBERS]
    return l2(np.concatenate(blocks, axis=1))


def encoder_matrix(name, member_embs):
    if name == "ens_sv4":
        return fuse_sv4(member_embs)
    return member_embs[name]


# ------------------------------------------------------------------------- run
def run(models, sanity=False, speaker_cap=None, dump_dir=None):
    rows = agency_segments()
    if speaker_cap is not None:
        keep_spk = sorted({r["speaker_id"] for r in rows})[:speaker_cap]
        keep_spk = set(keep_spk)
        rows = [r for r in rows if r["speaker_id"] in keep_spk]
    spk = np.array([r["speaker_id"] for r in rows])
    src = np.array([r["recording_source"] for r in rows])
    segids = [r["segment_id"] for r in rows]
    print(f"agency subset: {len(rows)} segments, {len(set(spk.tolist()))} speakers, "
          f"{len(set(src.tolist()))} recording sources")

    # base encoders we must actually embed
    base = []
    for m in models:
        base += SV4_MEMBERS if m == "ens_sv4" else [m]
    base = sorted(set(base))

    conditions = ["full", "loudnorm", "loud+tilt"]
    results = {m: {} for m in models}
    # audible/16k/loudness-shift verification artifacts
    verify = {}

    for cond in conditions:
        if cond == "full":
            # reuse cached raw embeddings, aligned to our agency subset by segment_id
            member = {}
            for e in base:
                d = np.load(EMB / f"{e}.npz", allow_pickle=True)
                pos = {s: i for i, s in enumerate(d["segment_id"].tolist())}
                idx = np.array([pos[s] for s in segids])
                member[e] = d["emb"][idx].astype(np.float32)
        else:
            print(f"  re-embedding condition '{cond}' with {base} ...", flush=True)
            member = embed_condition(rows, cond, base)
            NORM_EMB.mkdir(parents=True, exist_ok=True)
            for e in base:
                np.savez(NORM_EMB / f"{e}__{cond}.npz", emb=member[e],
                         segment_id=np.array(segids), speaker_id=spk, recording_source=src)
        for m in models:
            X = encoder_matrix(m, member)
            results[m][cond] = compute_metrics(X, spk, src)
            r = results[m][cond]
            print(f"    {m:9s} [{cond:9s}] misID {r['misid']*100:5.2f}%  hub {r['hubness_skew']:5.2f}  "
                  f"same-agency NN {r['nn_same_source_frac']*100:4.1f}% (chance {r['chance_same_source']*100:3.1f}%, "
                  f"enrich {r['same_source_enrichment']:.2f}x)", flush=True)

    # dump 3 verification WAVs (audible, 16k, loudness-shifted)
    if dump_dir is not None:
        dump_dir = Path(dump_dir); dump_dir.mkdir(parents=True, exist_ok=True)
        for r in rows[:3]:
            w = load_wav(r["segment_path"])
            wl = apply_condition(w, "loudnorm")
            wt = apply_condition(w, "loud+tilt")
            base_lufs = _METER.integrated_loudness(w.astype(np.float64))
            l_lufs = _METER.integrated_loudness(wl.astype(np.float64))
            t_lufs = _METER.integrated_loudness(wt.astype(np.float64))
            stem = r["segment_id"]
            sf.write(dump_dir / f"{stem}__orig.wav", w, SR)
            sf.write(dump_dir / f"{stem}__loudnorm.wav", wl, SR)
            sf.write(dump_dir / f"{stem}__loudtilt.wav", wt, SR)
            verify[stem] = {"orig_lufs": float(base_lufs), "loudnorm_lufs": float(l_lufs),
                            "loudtilt_lufs": float(t_lufs), "sr": SR,
                            "orig_rms": float(np.sqrt(np.mean(w**2))),
                            "loudnorm_rms": float(np.sqrt(np.mean(wl**2))),
                            "nonsilent": bool(np.max(np.abs(wl)) > 1e-3)}
            print(f"  VERIFY {stem}: LUFS {base_lufs:.1f} -> loud {l_lufs:.1f} / tilt {t_lufs:.1f}; "
                  f"16k, nonsilent={verify[stem]['nonsilent']}")

    out = {"target_lufs": TARGET_LUFS, "tilt_band_hz": [TILT_LO, TILT_HI],
           "n_segments": len(rows), "n_speakers": int(len(set(spk.tolist()))),
           "n_recording_sources": int(len(set(src.tolist()))),
           "sanity": sanity, "speaker_cap": speaker_cap,
           "results": results, "verify_files": verify}
    return out


def print_tables(out):
    print("\n=== TASK 1: mastering-normalized rerun (raw cosine, agency subset) ===")
    for metric, label, scale in [("misid", "misID %", 100),
                                 ("same_source_enrichment", "same-agency NN enrich x", 1),
                                 ("hubness_skew", "hubness skew", 1)]:
        print(f"\n-- {label} --")
        print(f"{'encoder':10s} {'full':>10s} {'loudnorm':>10s} {'loud+tilt':>10s}")
        for m, cds in out["results"].items():
            vals = []
            for c in ["full", "loudnorm", "loud+tilt"]:
                v = cds[c][metric]
                vals.append(f"{v*scale:10.2f}" if v is not None else f"{'-':>10s}")
            print(f"{m:10s} " + " ".join(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ecapa", "animeva", "ens_sv4"])
    ap.add_argument("--sanity", action="store_true",
                    help="one-encoder small-cap smoke test + dump 3 verification WAVs")
    ap.add_argument("--speaker-cap", type=int, default=None)
    args = ap.parse_args()

    if args.sanity:
        out = run(["ecapa"], sanity=True, speaker_cap=8,
                  dump_dir="output/audio_samples/mastering_norm")
        print_tables(out)
        AN.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(AN / "mastering_normalized_rerun_sanity.json", "w"), indent=2)
        print(f"\n-> {AN/'mastering_normalized_rerun_sanity.json'}")
        return

    out = run(args.models, speaker_cap=args.speaker_cap,
              dump_dir="output/audio_samples/mastering_norm")
    print_tables(out)
    AN.mkdir(parents=True, exist_ok=True)
    p = AN / "mastering_normalized_rerun.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

"""Scenario-1 base rate: how often does an INNOCENT amateur (general-public) voice
coincidentally match a voice actor? We score Common Voice JA (general public) clips
against the real-actor agency gallery (actors with >= 8 segments, per-actor
centroids) at the SAME real-VA-calibrated EER operating point used in the clone
probe, and report the false-match / wrongful-attribution rate.

Scoring runs in the shared centered cosine space (src/centered_gallery.py) --
the same space the threshold was calibrated in. (An earlier version applied the
centered-space threshold to uncentered cosines, which inflated the false-match
rate by a coordinate-system artifact.) Rates are reported at both the
same-file-genuine threshold (historical protocol) and the cross-file-genuine
threshold (deployment-realistic).

A clip "false-matches" if its top centered cosine to any gallery actor >= threshold
(the deployed operating point would flag it as "this is voice actor X"). We also
report which actors absorb the false matches (hub concentration) and how the
per-clip risk compounds over many clips (an uploader posts dozens -- cf. the
188-video case).

CHANNEL CAVEAT: Common Voice is crowd-mic (~24 dB) vs the studio gallery, so the
channel mismatch likely pushes amateurs APART from the gallery -> this is a
conservative (lower-bound-ish) estimate of coincidental resemblance.

Usage: python -m src.amateur_falsematch --models animeva ecapa
"""
from __future__ import annotations
import argparse, glob, json
from collections import Counter
from pathlib import Path
import numpy as np, torch, soundfile as sf
from src.centered_gallery import centered_gallery, center_query
from src.embeddings import build_extractor

CV = "/path/to/va-data/processed/cv"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--max-clips", type=int, default=2400)
    ap.add_argument("--clips-dir", default=CV, help="directory of 16 kHz query wavs (default: Common Voice JA)")
    ap.add_argument("--label", default="amateur", help="output name: output/analysis/{label}_falsematch.json")
    args = ap.parse_args()
    names_reg = {}
    for l in open("data/registry/speakers.jsonl", encoding="utf-8"):
        s = json.loads(l); names_reg[s["speaker_id"]] = s["name"]
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(args.clips_dir + "/*.wav"))
    if len(files) > args.max_clips:
        files = list(rng.choice(files, args.max_clips, replace=False))
    out = {}
    for model in args.models:
        ext = build_extractor(model, "cuda")
        gal = centered_gallery(model)
        Cn, mean, gnames = gal["Cn"], gal["mean"], gal["actor_ids"]
        thr, thr_xf = gal["thr"], gal["thr_xfile"]
        clip_spk, top, nn, collat = [], [], [], []
        for f in files:
            w, sr = sf.read(f, dtype="float32")
            assert sr == 16000, f"{f}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
            w = w.mean(1) if w.ndim > 1 else w
            v = center_query(np.asarray(ext.embed(torch.from_numpy(w)), np.float32), mean)
            sims = Cn @ v
            top.append(float(sims.max())); nn.append(gnames[int(sims.argmax())])
            collat.append(int((sims >= thr).sum()))
            clip_spk.append(Path(f).stem.split("__")[0])
        top = np.array(top); collat = np.array(collat); clip_spk = np.array(clip_spk)

        def summarize(t):
            fm = top >= t
            spk_ever = {s: bool(fm[clip_spk == s].any()) for s in set(clip_spk)}
            uspk = sorted(set(clip_spk)); rng2 = np.random.default_rng(1)
            boot = []
            for _ in range(2000):
                samp = rng2.choice(uspk, len(uspk), replace=True)
                boot.append(np.concatenate([fm[clip_spk == s] for s in samp]).mean())
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            hub = Counter(np.array(nn)[fm].tolist())
            return fm, {"threshold": float(t),
                        "false_match_per_clip": float(fm.mean()), "false_match_per_clip_ci95": ci,
                        "false_match_per_speaker_ever": float(np.mean(list(spk_ever.values()))),
                        "top_absorbing_actors": [(names_reg.get(a, a), c) for a, c in hub.most_common(5)]}

        fm, main_res = summarize(thr)
        _, xf_res = summarize(thr_xf)
        res = {"model": model, "n_clips": len(files), "n_amateurs": len(set(clip_spk.tolist())),
               "n_gallery": len(gnames), "space": "centered",
               **main_res,
               "mean_collateral_when_flagged": float(collat[fm].mean()) if fm.any() else 0.0,
               "at_crossfile_threshold": xf_res}
        out[model] = res
        print(f"\n#### {model} (gallery {len(gnames)} actors, thr {thr:.3f} / xfile {thr_xf:.3f}, "
              f"{len(files)} amateur clips / {res['n_amateurs']} speakers) ####")
        for tag, r in [("same-file thr", main_res), ("cross-file thr", xf_res)]:
            print(f"  [{tag}] per-CLIP false match: {r['false_match_per_clip']*100:.2f}% "
                  f"CI[{r['false_match_per_clip_ci95'][0]*100:.2f}-{r['false_match_per_clip_ci95'][1]*100:.2f}]  "
                  f"per-SPEAKER ever: {r['false_match_per_speaker_ever']*100:.1f}%")
        p = main_res["false_match_per_clip"]
        print(f"  over 188 clips P(>=1 flag) = {(1-(1-p)**188)*100:.1f}%  (if per-clip indep.)")
        print(f"  mean collateral when flagged: {res['mean_collateral_when_flagged']:.2f} actors")
        print(f"  actors absorbing false matches: {main_res['top_absorbing_actors']}")
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    outp = f"output/analysis/{args.label}_falsematch.json"
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()

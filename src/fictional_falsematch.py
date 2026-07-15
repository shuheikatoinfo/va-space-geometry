"""Do genuinely FICTIONAL voices (Irodori VoiceDesign, caption-conditioned, no
reference person) coincidentally match a real voice actor? This is the
provenance-cleaner version of the enclosure measurement -- no JVS VA-overlap
caveat (these are designed novel speakers; note the generator's own training
data is undisclosed, so training-set independence from the matched actors
cannot be guaranteed). We also check the cos LEVEL of the matches vs the
genuine same-speaker baseline: matches well below same-speaker level =
'similar but not a copy', rebutting the "a TTS voice is just a copy of
training data" objection.

Scoring runs in the shared centered cosine space (src/centered_gallery.py),
the same space the EER threshold was calibrated in; the genuine same-speaker
baseline uses held-out genuine trials (never the segments that built the
centroid), so the baseline is not self-inflated.

Usage: python -m src.fictional_falsematch --models animeva ecapa
"""
from __future__ import annotations
import argparse, glob
from collections import Counter
from pathlib import Path
import numpy as np, torch, soundfile as sf, librosa
from src.centered_gallery import centered_gallery, center_query
from src.embeddings import build_extractor
import json

FICT = "/path/to/va-data/clones/voicedesign_fictional"
names_reg = {}
for _l in open("data/registry/speakers.jsonl", encoding="utf-8"):
    _s = json.loads(_l); names_reg[_s["speaker_id"]] = _s["name"]


def emb16k(ext, f):
    w, sr = sf.read(f, dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
    if sr != 16000:
        w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    return np.asarray(ext.embed(torch.from_numpy(w.astype(np.float32))), np.float32).reshape(-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    args = ap.parse_args()
    files = sorted(glob.glob(FICT + "/*.wav"))
    print(f"fictional voices: {len(files)}")
    out = {}
    for model in args.models:
        ext = build_extractor(model, "cuda")
        gal = centered_gallery(model)
        Cn, mean, gn = gal["Cn"], gal["mean"], gal["actor_ids"]
        thr, thr_xf = gal["thr"], gal["thr_xfile"]
        # genuine same-speaker baseline: held-out genuine scores (clip vs own
        # centroid, centered space, query never inside the centroid). The
        # cross-file operating point is compared against the cross-file genuine
        # distribution (same protocol as its threshold), not the same-session one.
        same = gal["gen_scores"]
        same_p10 = float(np.percentile(same, 10))
        same_xf = gal["gen_scores_xfile"]
        top, nn, caps = [], [], []
        for f in files:
            v = center_query(emb16k(ext, f), mean)
            sims = Cn @ v
            top.append(float(sims.max())); nn.append(gn[int(sims.argmax())])
            caps.append(Path(f).stem.split("_")[0])  # cap<k> — the design prompt group
        top = np.array(top); caps = np.array(caps)

        def boot_ci(t, n_boot=1000):
            # bootstrap over caption groups (the 4 sampled candidates within a
            # caption share the design prompt and seed; see gen_voicedesign_cli.sh)
            ucaps = np.unique(caps)
            rng = np.random.default_rng(0)
            rates = []
            for _ in range(n_boot):
                samp = rng.choice(ucaps, len(ucaps), replace=True)
                mask = np.concatenate([np.flatnonzero(caps == c) for c in samp])
                rates.append(float(np.mean(top[mask] >= t)))
            return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]

        def summarize(t, gen):
            fm = top >= t
            gen_p10 = float(np.percentile(gen, 10))
            below = float(np.mean(top[fm] < gen_p10)) if fm.any() else 0.0
            # where each flagged match sits INSIDE the genuine same-speaker score
            # distribution (mean percentile; 0.5 = indistinguishable from a real
            # same-speaker pair, low = clearly weaker than a genuine match)
            pct = float(np.mean([np.mean(gen <= s) for s in top[fm]])) if fm.any() else float("nan")
            return {"threshold": float(t), "false_match_rate": float(fm.mean()),
                    "n_matches": int(fm.sum()),
                    "match_cos_mean": float(top[fm].mean()) if fm.any() else 0.0,
                    "flagged_below_genuine_p10_frac": below,
                    "match_mean_pctile_in_genuine": pct,
                    "top_actors": [(names_reg.get(a, a), c)
                                   for a, c in Counter(np.array(nn)[fm].tolist()).most_common(5)]}

        main_res = summarize(thr, same); main_res["ci95_caption_boot"] = boot_ci(thr)
        xf_res = summarize(thr_xf, same_xf); xf_res["ci95_caption_boot"] = boot_ci(thr_xf)
        res = {"model": model, "n_fictional": len(files), "space": "centered",
               "rows": [{"file": Path(f).name, "cap": str(c), "top1": float(t),
                         "nn_actor": str(a), "fm": int(t >= thr), "fm_xf": int(t >= thr_xf)}
                        for f, c, t, a in zip(files, caps, top, nn)],
               **main_res,
               "match_cos_max": float(top.max()),
               "genuine_same_speaker_mean": float(same.mean()),
               "genuine_same_speaker_p10": same_p10,
               "at_crossfile_threshold": xf_res}
        out[model] = res
        print(f"\n#### {model} (thr {thr:.3f} / xfile {thr_xf:.3f}, {len(files)} fictional voices) ####")
        for tag, r in [("same-file thr", main_res), ("cross-file thr", xf_res)]:
            print(f"  [{tag}] FALSE-MATCH rate: {r['false_match_rate']*100:.1f}%  ({r['n_matches']}/{len(files)})"
                  f"  below-genuine-p10: {r['flagged_below_genuine_p10_frac']*100:.0f}%")
        print(f"  match cos: mean {main_res['match_cos_mean']:.3f}  max {res['match_cos_max']:.3f}"
              f"  | genuine same-spk mean {same.mean():.3f} p10 {same_p10:.3f}")
        print(f"  top matched actors: {main_res['top_actors']}")
    # Cross-encoder consistency (memorization test): regurgitation of a memorized
    # actor predicts both encoders converge on the SAME actor for a flagged voice;
    # coincidental geometry predicts disagreement.
    if len(out) >= 2:
        ms = list(out)
        rows = {m: {r["file"]: r for r in out[m]["rows"]} for m in ms}
        common = sorted(set(rows[ms[0]]) & set(rows[ms[1]]))
        agree_all = float(np.mean([rows[ms[0]][f]["nn_actor"] == rows[ms[1]][f]["nn_actor"]
                                   for f in common]))
        both_fm = [f for f in common if rows[ms[0]][f]["fm"] and rows[ms[1]][f]["fm"]]
        agree_fm = (float(np.mean([rows[ms[0]][f]["nn_actor"] == rows[ms[1]][f]["nn_actor"]
                                   for f in both_fm])) if both_fm else float("nan"))
        out["cross_encoder"] = {"models": ms, "n_common": len(common),
                                "nn_agreement_all": agree_all,
                                "n_flagged_by_both": len(both_fm),
                                "nn_agreement_flagged_by_both": agree_fm}
        print(f"\ncross-encoder NN agreement: all {agree_all*100:.1f}%  "
              f"flagged-by-both n={len(both_fm)} agreement "
              f"{agree_fm*100 if both_fm else float('nan'):.1f}%")
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/fictional_falsematch.json", "w"), ensure_ascii=False, indent=2)
    print("\nwrote output/analysis/fictional_falsematch.json")


if __name__ == "__main__":
    main()

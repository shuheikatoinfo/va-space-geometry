"""Contamination audit: do animeva's training-set speakers (246/1,168 'seen',
name-matched vs a public VNDB voice-actor dump -> data/registry/animeva_seen.json)
drive its results?

Computes, from the per-segment verification margins already saved by analyze.py
(output/analysis/{model}.npz):
  1. per-speaker misID (fraction of the speaker's queries with margin < 0),
     split seen vs unseen, with speaker-level bootstrap CIs;
  2. the gender misID gap restricted to HELD-OUT (unseen) speakers, same
     bootstrap protocol as fairness_analysis.py.

This gives the abstract's "contamination does not explain the results" claim a
reproducible artifact (previously the numbers existed only in
docs/rigorous_evaluation.md).

Usage: python -m src.contamination_audit --models animeva ecapa
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

AN = Path("output/analysis")
FEM = {"女", "female", "F", "f", "woman"}
MALE = {"男", "male", "M", "m", "man"}


def boot_ci(vals, reps=2000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(reps)])
    return float(vals.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    args = ap.parse_args()
    seen_map = json.load(open("data/registry/animeva_seen.json", encoding="utf-8"))
    gender_map = json.load(open("data/registry/gender.json", encoding="utf-8"))
    out = {}
    for model in args.models:
        an = np.load(AN / f"{model}.npz", allow_pickle=True)
        vm = an["verification_margin"].astype(np.float32)
        has = an["has_same"].astype(bool)
        spk = an["speaker_id"].astype(str)
        rows = []
        for s in sorted(set(spk.tolist())):
            m = vm[(spk == s) & has]
            if m.size == 0:
                continue
            rows.append({"spk": s, "misid": float((m < 0).mean()),
                         "seen": bool(seen_map.get(s, False)),
                         "gender": gender_map.get(s, "unknown")})
        res = {"n_speakers": len(rows)}
        for tag, sel in [("seen", [r for r in rows if r["seen"]]),
                         ("unseen", [r for r in rows if not r["seen"]])]:
            mean, ci, _ = boot_ci([r["misid"] for r in sel], seed=0)
            res[tag] = {"n_spk": len(sel), "misid": mean, "misid_ci95": ci}
        # gender gap on held-out speakers only
        fem = [r["misid"] for r in rows if not r["seen"] and r["gender"] in FEM]
        male = [r["misid"] for r in rows if not r["seen"] and r["gender"] in MALE]
        if len(fem) >= 5 and len(male) >= 5:
            fmean, fci, bf = boot_ci(fem, seed=1)
            mmean, mci, bm = boot_ci(male, seed=2)
            gap = bf - bm
            res["heldout_gender"] = {
                "n_female": len(fem), "n_male": len(male),
                "female_misid": fmean, "female_ci95": fci,
                "male_misid": mmean, "male_ci95": mci,
                "gap": fmean - mmean,
                "gap_ci95": [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))],
            }
        out[model] = res
        print(f"#### {model} contamination audit ({len(rows)} speakers) ####")
        for tag in ["seen", "unseen"]:
            r = res[tag]
            print(f"  {tag:6s} n={r['n_spk']:4d}  misID={r['misid']*100:5.2f}%  CI[{r['misid_ci95'][0]*100:.2f}-{r['misid_ci95'][1]*100:.2f}]")
        g = res.get("heldout_gender")
        if g:
            print(f"  held-out gender: F {g['female_misid']*100:.1f}% (n={g['n_female']})  "
                  f"M {g['male_misid']*100:.1f}% (n={g['n_male']})  gap {g['gap']*100:+.1f} pp "
                  f"CI[{g['gap_ci95'][0]*100:.1f},{g['gap_ci95'][1]*100:.1f}]")
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "contamination_audit.json", "w"), ensure_ascii=False, indent=2)
    print("wrote output/analysis/contamination_audit.json")


if __name__ == "__main__":
    main()

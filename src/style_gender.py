"""Style-stratified gender misID (reviewer: is the gender gap a style-composition
artifact?). Uses precomputed per-segment margins; no GPU.

For each encoder we compute per-speaker misID WITHIN each style category, then the
female-vs-male gap within narration and within dialogue separately, with a
speaker-level bootstrap CI on each within-style gap. If the gap survives within a
single style, it is not merely a female/male difference in style composition.

Usage: python -m src.style_gender --models ecapa animeva
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

from src.style_analysis import style_cat

EMB = Path("output/embeddings"); AN = Path("output/analysis")
FEM = {"女", "female", "F", "f", "woman"}; MALE = {"男", "male", "M", "m", "man"}


def boot_gap(fem, male, reps=2000, seed=0):
    if len(fem) < 5 or len(male) < 5:
        return None
    fem, male = np.array(fem), np.array(male); rng = np.random.default_rng(seed)
    bf = np.array([rng.choice(fem, len(fem), True).mean() for _ in range(reps)])
    bm = np.array([rng.choice(male, len(male), True).mean() for _ in range(reps)])
    gap = bf - bm
    return {"n_female": len(fem), "n_male": len(male),
            "female_misid": float(fem.mean()), "male_misid": float(male.mean()),
            "gap": float(fem.mean() - male.mean()),
            "gap_ci95": [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))],
            "gap_p_onesided": float(np.mean(gap <= 0))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    args = ap.parse_args()
    gender = json.load(open("data/registry/gender.json", encoding="utf-8"))
    out = {}
    for model in args.models:
        d = np.load(EMB / f"{model}.npz", allow_pickle=True)
        spk = np.asarray(d["speaker_id"]); style = np.array([style_cat(x) for x in d["style_label"]])
        an = np.load(AN / f"{model}.npz", allow_pickle=True)
        vm = an["verification_margin"].astype(np.float32); has = an["has_same"].astype(bool)
        # per-speaker-per-style misID
        cell = defaultdict(list)  # (spk, style) -> margins
        for i in range(len(spk)):
            if has[i]:
                cell[(str(spk[i]), style[i])].append(vm[i])
        # aggregate: for each style, list of per-speaker misID split by gender
        res = {}
        for st in ["narration", "dialogue", "name", "other", "full", "freeform"]:
            fem, male = [], []
            for (s, c), ms in cell.items():
                if c != st:
                    continue
                mis = float((np.array(ms) < 0).mean()); g = gender.get(s, "unknown")
                if g in FEM: fem.append(mis)
                elif g in MALE: male.append(mis)
            gp = boot_gap(fem, male)
            if gp:
                res[st] = gp
        # overall (all styles pooled per speaker) for reference
        femA, maleA = [], []
        byspk = defaultdict(list)
        for i in range(len(spk)):
            if has[i]:
                byspk[str(spk[i])].append(vm[i])
        for s, ms in byspk.items():
            mis = float((np.array(ms) < 0).mean()); g = gender.get(s, "unknown")
            if g in FEM: femA.append(mis)
            elif g in MALE: maleA.append(mis)
        res["_overall"] = boot_gap(femA, maleA)
        out[model] = res
        print(f"\n#### {model}: gender misID gap stratified by style ####")
        for st, gp in res.items():
            if gp is None: continue
            print(f"  {st:10s}  F {gp['female_misid']*100:5.1f}% (n={gp['n_female']:3d})  "
                  f"M {gp['male_misid']*100:5.1f}% (n={gp['n_male']:3d})  "
                  f"gap {gp['gap']*100:+5.1f}pp  CI[{gp['gap_ci95'][0]*100:+.1f},{gp['gap_ci95'][1]*100:+.1f}]  p={gp['gap_p_onesided']:.3f}")
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "style_gender.json", "w"), ensure_ascii=False, indent=2)
    print("\n-> output/analysis/style_gender.json")


if __name__ == "__main__":
    main()

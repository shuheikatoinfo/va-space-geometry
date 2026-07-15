"""Story B: is protection unequal? Are thin margins / misidentification / wrongful
attribution concentrated in subgroups (gender, fame, agency)?

Attributes:
  - gender: scraped labels from agency listing pages + an agency listing API
    (data/registry/gender.json). Using scraped labels rather than an
    F0-derived split avoids circularity with the F0 content of the embedding
    itself; note the labels cover only the ~458 speakers whose agencies publish
    gendered rosters (an agency-selected, not random, subsample).
  - fame: number of titles in the registry 'works' field (出演作品) as a seniority/
    prominence proxy
  - agency: recording_source

Per-speaker outcome = mean verification margin + misidentification rate (from the
full-set analysis npz), grouped/regressed by attribute.

Usage: python -m src.fairness_analysis --model ecapa
"""
from __future__ import annotations

import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB = Path("output/embeddings"); AN = Path("output/analysis")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ecapa"); args = ap.parse_args()
    # registry: works-count + scraped gender
    works = {}
    for l in open("data/registry/speakers.jsonl", encoding="utf-8"):
        s = json.loads(l)
        works[s["speaker_id"]] = len(re.findall(r"[『「][^』」]+[』」]", s.get("works", "")))
    gender_map = json.load(open("data/registry/gender.json", encoding="utf-8"))
    # per-segment margins + speaker + source + paths
    d = np.load(EMB / f"{args.model}.npz", allow_pickle=True)
    spk = np.asarray(d["speaker_id"]); src = np.asarray(d["recording_source"]); seg = np.asarray(d["segment_id"])
    an = np.load(AN / f"{args.model}.npz", allow_pickle=True)
    vm = an["verification_margin"].astype(np.float32); has = an["has_same"]
    seg2path = {}
    for l in open("data/processed/segments.jsonl", encoding="utf-8"):
        if '"segment_id"' in l:
            r = json.loads(l); seg2path[r["segment_id"]] = r["segment_path"]

    by = defaultdict(list)
    for i in range(len(spk)):
        if has[i]:
            by[spk[i]].append(i)
    rows = []
    for s, idx in by.items():
        m = vm[idx]
        rows.append({"spk": s, "agency": str(src[idx[0]]), "gender": gender_map.get(s, "unknown"),
                     "works": works.get(s, 0), "n": len(idx), "mean_margin": float(m.mean()),
                     "misid": float((m < 0).mean()), "critical": float((np.abs(m) < 0.02).mean())})
    thr = 0.0

    def group(key, buckets=None):
        g = defaultdict(list)
        for r in rows:
            k = r[key] if buckets is None else buckets(r)
            g[k].append(r)
        return {k: {"n_spk": len(v), "mean_margin": float(np.mean([x["mean_margin"] for x in v])),
                    "misid": float(np.mean([x["misid"] for x in v])),
                    "critical": float(np.mean([x["critical"] for x in v]))} for k, v in g.items()}

    wq = np.quantile([r["works"] for r in rows], [0.5])

    # speaker-level bootstrap CI for the gender misID gap (resample speakers within
    # each group, 2000 reps, seeded). Reports per-group misID CI, the F-M gap CI, and
    # a one-sided bootstrap p-value (fraction of reps with gap <= 0).
    def gender_gap_ci(reps=2000):
        fem_keys = {"女", "female", "F", "f", "woman"}
        male_keys = {"男", "male", "M", "m", "man"}
        fem = [r["misid"] for r in rows if r["gender"] in fem_keys]
        male = [r["misid"] for r in rows if r["gender"] in male_keys]
        if len(fem) < 5 or len(male) < 5:
            return None
        fem, male = np.array(fem), np.array(male)
        rng = np.random.default_rng(0)
        bf = np.array([rng.choice(fem, len(fem), replace=True).mean() for _ in range(reps)])
        bm = np.array([rng.choice(male, len(male), replace=True).mean() for _ in range(reps)])
        gap = bf - bm
        return {
            "n_female": int(len(fem)), "n_male": int(len(male)),
            "female_misid": float(fem.mean()),
            "female_ci95": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
            "male_misid": float(male.mean()),
            "male_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "gap": float(fem.mean() - male.mean()),
            "gap_ci95": [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))],
            "gap_p_onesided": float(np.mean(gap <= 0)),
        }

    out = {
        "model": args.model, "n_speakers": len(rows), "f0_split_hz": thr,
        "by_gender": group("gender"),
        "gender_gap_ci": gender_gap_ci(),
        "by_fame": group(None, lambda r: "few_works(<=med)" if r["works"] <= wq[0] else "many_works(>med)"),
        "by_agency_top": {},
    }
    # top agencies by speaker count
    ag = defaultdict(list)
    for r in rows:
        ag[r["agency"]].append(r)
    for a, v in sorted(ag.items(), key=lambda kv: -len(kv[1]))[:8]:
        out["by_agency_top"][a] = {"n_spk": len(v),
            "mean_margin": float(np.mean([x["mean_margin"] for x in v])),
            "misid": float(np.mean([x["misid"] for x in v]))}

    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / f"fairness_{args.model}.json", "w"), ensure_ascii=False, indent=2)
    print(f"#### {args.model} fairness (n={len(rows)} speakers, F0 split {thr:.0f}Hz) ####")
    for grp in ["by_gender", "by_fame"]:
        print(f"  {grp}:")
        for k, v in out[grp].items():
            print(f"    {k:18s} n={v['n_spk']:4d}  mean_margin={v['mean_margin']:+.3f}  misid={v['misid']*100:4.1f}%  crit={v['critical']*100:4.1f}%")
    g = out.get("gender_gap_ci")
    if g:
        print("  gender misID (speaker bootstrap 95% CI):")
        print(f"    female  {g['female_misid']*100:4.1f}%  CI[{g['female_ci95'][0]*100:.1f}-{g['female_ci95'][1]*100:.1f}]  (n={g['n_female']})")
        print(f"    male    {g['male_misid']*100:4.1f}%  CI[{g['male_ci95'][0]*100:.1f}-{g['male_ci95'][1]*100:.1f}]  (n={g['n_male']})")
        print(f"    F-M gap {g['gap']*100:+.1f}pp  CI[{g['gap_ci95'][0]*100:+.1f},{g['gap_ci95'][1]*100:+.1f}]  one-sided p={g['gap_p_onesided']:.4f}")
    print("  by_agency (top, mean_margin asc = thinnest first):")
    for a, v in sorted(out["by_agency_top"].items(), key=lambda kv: kv[1]["mean_margin"]):
        print(f"    {a[:22]:22s} n={v['n_spk']:3d}  margin={v['mean_margin']:+.3f}  misid={v['misid']*100:4.1f}%")
    print(f"-> {AN/'fairness_analysis.json'}")


if __name__ == "__main__":
    main()

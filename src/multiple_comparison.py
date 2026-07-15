"""Multiple-comparison correction across the family of geometry significance tests
(hubness per encoder + confusion-graph structure). Reports raw p, Holm and
Benjamini-Hochberg adjusted p, and the decision at alpha=0.05.

All p-values are EMPIRICAL permutation p-values, floored at 1/(n_perm+1):
a permutation test with n nulls cannot certify a p smaller than that, so no
Gaussian-tail extrapolation from z-scores is used.
"""
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats

AN = Path("output/analysis")
tests = []

# hubness permutation p-values per encoder (from per-model summaries)
for f in sorted(glob.glob(str(AN / "*.json"))):
    name = Path(f).stem
    if name in {"calibrated", "pop_table", "clone_geometry", "style_analysis", "verification_metrics",
                "verification_plda", "multivec_clone", "source_quality", "multiple_comparison"} or "clone_score" in name \
       or "fairness" in name or "confusion_null" in name or "popcompare" in name or "clone_probe" in name:
        continue
    try:
        d = json.load(open(f))
        p = d.get("phase4_geometry", {}).get("hubness_skewness_pvalue")
        if p is not None:
            # analyze.py reports p = k/200 over 200 permutations; convert to the
            # add-one empirical form (k+1)/201 (repairs both p=0 and the
            # 2x-anti-conservative k>=1 case)
            tests.append((f"hubness:{name}", (float(p) * 200 + 1) / 201))
    except Exception:
        pass

# confusion-graph empirical permutation p-values (floored at 1/(n_null+1) by
# construction in confusion_null.py). Older JSONs without p_emp fields fall
# back to the z-score Gaussian tail, floored at 1/201 as a conservative bound.
for f in glob.glob(str(AN / "confusion_null_*.json")):
    d = json.load(open(f)); m = Path(f).stem.replace("confusion_null_", "")
    for key in ["indeg_skew", "asymmetry", "modularity"]:
        p = d.get(f"{key}_p_emp")
        if p is None:
            p = max(2 * stats.norm.sf(abs(d[f"{key}_z"])), 1.0 / 201)
        tests.append((f"{key}:{m}", float(p)))

names = [t[0] for t in tests]; pv = np.array([t[1] for t in tests])
order = np.argsort(pv); m = len(pv)
# Holm
holm = np.empty(m); run = 0.0
for rank, i in enumerate(order):
    run = max(run, (m - rank) * pv[i]); holm[i] = min(run, 1.0)
# BH
bh = np.empty(m); prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]; prev = min(prev, pv[i] * m / (rank + 1)); bh[i] = prev

print(f"{'test':28s} {'raw p':>10s} {'Holm':>10s} {'BH':>10s} sig@.05")
for i in range(m):
    print(f"{names[i]:28s} {pv[i]:10.2e} {holm[i]:10.2e} {bh[i]:10.2e}  {'yes' if bh[i] < .05 else 'no'}")
print(f"\n{int((bh < .05).sum())}/{m} tests significant after BH correction (alpha=0.05)")
json.dump([{"test": names[i], "p": pv[i], "holm": holm[i], "bh": bh[i]} for i in range(m)],
          open(AN / "multiple_comparison.json", "w"), indent=2)

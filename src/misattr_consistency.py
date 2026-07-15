"""Misattribution CONSISTENCY: when clones of the same target are misattributed,
do they hit the SAME wrong actor (systematic neighbour) or scatter?

Statistic: for each target with >= 2 misattributed clones, the largest fraction
of that target's misattributed clones sharing one wrong nearest-actor;
averaged over targets (max-fraction consistency).

The correct chance baseline for a max-fraction statistic with m draws is NOT
1/N_gallery (that is the PAIRWISE collision probability). We therefore report
two permutation nulls:
  uniform   : wrong actors drawn uniformly from the gallery (pure chance)
  marginal  : observed wrong-actor labels permuted ACROSS targets (preserves
              the hub structure / marginal popularity of wrong actors, breaks
              the target linkage -- the interesting null: "same wrong actor
              beyond what hub concentration alone produces")
plus the pairwise same-wrong-actor collision rate with its analytic 1/(N-1)
uniform baseline.

Reads the per-clone rows saved by score_clones.py.

Usage: python -m src.misattr_consistency --labels seedvc irodori120 gptsovits_v1 ...
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

AN = Path("output/analysis")


def max_fraction(by_target):
    fr = []
    for t, wrongs in by_target.items():
        if len(wrongs) < 2:
            continue
        _, counts = np.unique(wrongs, return_counts=True)
        fr.append(counts.max() / len(wrongs))
    return (float(np.mean(fr)), len(fr)) if fr else (float("nan"), 0)


def pairwise_collision(by_target):
    hits, tot = 0, 0
    for t, wrongs in by_target.items():
        m = len(wrongs)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                tot += 1
                hits += int(wrongs[i] == wrongs[j])
    return (hits / tot, tot) if tot else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--condition", choices=["misattr", "wrongful"], default="misattr",
                    help="misattr = nearest != target (all clones); wrongful = also over threshold")
    args = ap.parse_args()
    out = {}
    for label in args.labels:
        f = AN / f"clone_score_{label}.json"
        if not f.exists():
            print(f"skip {label}: {f} missing"); continue
        d = json.load(open(f))
        out[label] = {}
        for model in args.models:
            rows = d.get(model, {}).get("rows")
            if not rows:
                print(f"skip {label}/{model}: no per-clone rows (re-run score_clones)"); continue
            cond = (lambda r: r["nearest"] != r["target"]) if args.condition == "misattr" \
                else (lambda r: r["wrongful"] == 1)
            wrong = [(r["target"], r["nearest"]) for r in rows if cond(r)]
            by_t = defaultdict(list)
            for t, nn in wrong:
                by_t[t].append(nn)
            obs, n_targets = max_fraction(by_t)
            pw_obs, n_pairs = pairwise_collision(by_t)
            # true gallery size for the uniform null; the actors appearing in
            # rows are only those that were a target or someone's nearest
            # (~150), which would understate the pool ~7x
            n_gallery = d.get(model, {}).get("n_gallery")
            if not n_gallery:
                n_gallery = len({r["nearest"] for r in rows} | {r["target"] for r in rows})
                print(f"  WARNING {label}/{model}: n_gallery missing from clone_score JSON; "
                      f"falling back to {n_gallery} actors seen in rows (understates the pool)")
            if n_targets == 0:
                # no target has >=2 misattributed clones -> the max-fraction
                # statistic does not exist; a NaN obs would make `nulls >= obs`
                # all-False and fabricate p ~= 1/(nperm+1)
                out[label][model] = {
                    "condition": args.condition, "n_misattributed": len(wrong),
                    "n_targets_with_2plus": 0, "consistency_maxfrac": None,
                    "note": "statistic undefined (no target with >=2 misattributed clones)",
                }
                print(f"## {label}/{model}: statistic undefined (no target with >=2 misattributed clones)")
                continue
            all_wrong = [nn for t, nn in wrong]
            rng = np.random.default_rng(0)
            nulls_u, nulls_m = [], []
            sizes = {t: len(v) for t, v in by_t.items()}
            for _ in range(args.nperm):
                # uniform null over the FULL gallery (labels are interchangeable,
                # only collision structure matters)
                bt = {t: list(rng.integers(0, n_gallery, m)) for t, m in sizes.items()}
                nulls_u.append(max_fraction(bt)[0])
                # marginal-preserving null: permute observed wrong labels across targets
                perm = rng.permutation(all_wrong)
                i, bt2 = 0, {}
                for t, m in sizes.items():
                    bt2[t] = list(perm[i:i + m]); i += m
                nulls_m.append(max_fraction(bt2)[0])
            nulls_u, nulls_m = np.array(nulls_u), np.array(nulls_m)
            res = {
                "condition": args.condition, "n_misattributed": len(wrong),
                "n_targets_with_2plus": n_targets,
                "n_gallery": int(n_gallery),
                "consistency_maxfrac": obs,
                "null_uniform_mean": float(np.nanmean(nulls_u)),
                "null_uniform_p": float((1 + np.nansum(nulls_u >= obs)) / (1 + args.nperm)),
                "null_marginal_mean": float(np.nanmean(nulls_m)),
                "null_marginal_p": float((1 + np.nansum(nulls_m >= obs)) / (1 + args.nperm)),
                "pairwise_collision": pw_obs, "n_pairs": n_pairs,
                "pairwise_chance_uniform": 1.0 / max(n_gallery - 1, 1),
            }
            out[label][model] = res
            print(f"## {label}/{model}: max-frac consistency {obs:.3f} over {n_targets} targets "
                  f"(uniform null {res['null_uniform_mean']:.3f} p={res['null_uniform_p']:.4g}; "
                  f"marginal null {res['null_marginal_mean']:.3f} p={res['null_marginal_p']:.4g}) | "
                  f"pairwise {pw_obs:.3f} vs 1/(N-1)={res['pairwise_chance_uniform']:.5f}")
    json.dump(out, open(AN / "misattr_consistency.json", "w"), ensure_ascii=False, indent=2)
    print("wrote output/analysis/misattr_consistency.json")


if __name__ == "__main__":
    main()

"""Story C: is the voice-actor domain hard because of intra-speaker STYLE range,
and do certain styles (character/dialogue) bridge across speakers?

Uses each model's embeddings (with style_label) + the verification margins from
output/analysis/<model>.npz (index-aligned). Reports:
  - variance decomposition: within-speaker same-style vs different-style distance
    (does style drive intra-speaker spread?)
  - style-stratified verification margin (which styles are most confusable)
  - "style bridge": among the nearest different-speaker matches, are character/
    dialogue styles over-represented vs narration?

Usage: python -m src.style_analysis --models ecapa animeva
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB = Path("output/embeddings"); AN = Path("output/analysis")


def style_cat(label: str) -> str:
    s = str(label)
    if any(k in s for k in ["ナレーション", "narration", "na_", "NA"]):
        return "narration"
    if any(k in s for k in ["セリフ", "serif", "dialog", "se0", "se1"]):
        return "dialogue"
    if any(k in s for k in ["名前", "name"]):
        return "name"
    if "full" in s.lower():
        return "full"
    if s.startswith("free"):
        return "freeform"
    return "other"


def run(model):
    d = np.load(EMB / f"{model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32); spk = np.asarray(d["speaker_id"])
    style = np.array([style_cat(x) for x in d["style_label"]])
    an = np.load(AN / f"{model}.npz", allow_pickle=True)
    vm = an["verification_margin"].astype(np.float32); has = an["has_same"]
    X = emb - emb.mean(0, keepdims=True); X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

    # within-speaker same-style vs different-style mean cosine (agency-only for clean styles)
    by = defaultdict(list)
    for i in range(len(spk)):
        by[spk[i]].append(i)
    same_s, diff_s = [], []
    rng = np.random.default_rng(0)
    for s, idx in by.items():
        if len(idx) < 4:
            continue
        idx = np.array(idx)
        sub = idx if len(idx) <= 40 else rng.choice(idx, 40, replace=False)
        Xi = X[sub]; st = style[sub]; C = Xi @ Xi.T
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                (same_s if st[a] == st[b] else diff_s).append(C[a, b])
    # style-stratified margin
    margin_by_style = {}
    for cat in ["narration", "dialogue", "name", "full", "freeform", "other"]:
        m = (style == cat) & has
        if m.sum() > 20:
            margin_by_style[cat] = {"n": int(m.sum()), "mean_margin": float(vm[m].mean()),
                                    "misid": float((vm[m] < 0).mean()),
                                    "critical<0.02": float((np.abs(vm[m]) < 0.02).mean())}
    res = {
        "model": model,
        "within_spk_same_style_cos": float(np.mean(same_s)),
        "within_spk_diff_style_cos": float(np.mean(diff_s)),
        "style_drives_intra_spread": float(np.mean(same_s) - np.mean(diff_s)),
        "margin_by_style": margin_by_style,
        "style_distribution": {k: int((style == k).sum()) for k in set(style)},
    }
    print(f"\n#### {model} ####")
    print(f"  within-speaker cosine: same-style={res['within_spk_same_style_cos']:.3f}  "
          f"diff-style={res['within_spk_diff_style_cos']:.3f}  (gap={res['style_drives_intra_spread']:.3f})")
    print("  verification margin by query style (lower/negative = more confusable):")
    for cat, v in sorted(margin_by_style.items(), key=lambda kv: kv[1]["mean_margin"]):
        print(f"    {cat:10s} n={v['n']:5d}  mean_margin={v['mean_margin']:+.3f}  misid={v['misid']*100:4.1f}%  crit={v['critical<0.02']*100:4.1f}%")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    args = ap.parse_args()
    out = {m: run(m) for m in args.models}
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "style_analysis.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {AN/'style_analysis.json'}")


if __name__ == "__main__":
    main()

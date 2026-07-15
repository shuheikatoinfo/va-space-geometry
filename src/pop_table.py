"""Dump the matched VA-vs-control comparison (AS-norm) to JSON for the figures.

For each model, compares VA against each available control at a matched
(speakers x segments) size, mean over seeds. Output: output/analysis/pop_table.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.population_compare import load_pop, match_subset, asnorm_eval

MODELS = ["ecapa", "jxvector", "animeva"]
# (control pop, segs/speaker, n_speakers) — matched to control's available data
CONTROLS = [("jvs", 8, 100), ("jvsv", 8, 100), ("cv", 3, 455)]


def main():
    out = {}
    for model in MODELS:
        va = load_pop(model, "va")
        out[model] = {}
        for ctrl, m_seg, n_spk in CONTROLS:
            try:
                cp = load_pop(model, ctrl)
            except FileNotFoundError:
                continue
            cap = min(n_spk, (np.array(list(Counter(cp[1].tolist()).values())) >= m_seg).sum())
            cap = int(cap)
            rec = {}
            for tag, pop in [("VA", va), (ctrl, cp)]:
                accs = defaultdict(list)
                for sd in range(3):
                    e, s = match_subset(*pop, cap, m_seg, seed=sd)
                    r = asnorm_eval(e, s)
                    for k in ["rank1", "eer", "misid", "critical", "hub"]:
                        accs[k].append(r[k])
                rec[tag] = {k: float(np.mean(v)) for k, v in accs.items()}
            out[model][ctrl] = {"matched_speakers": cap, "segs_per_speaker": m_seg, **rec}
            print(f"{model} vs {ctrl}: VA EER {rec['VA']['eer']*100:.1f}% | {ctrl} EER {rec[ctrl]['eer']*100:.1f}%")
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/pop_table.json", "w"), ensure_ascii=False, indent=2)
    print("-> output/analysis/pop_table.json")


if __name__ == "__main__":
    main()

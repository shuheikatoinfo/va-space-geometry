"""Story D: the cost of a single fixed operating point vs per-speaker adaptive
thresholds and an abstain (reject) option.

From the per-query genuine (same_best) / impostor (diff_best) scores:
  - per-speaker EER thresholds: how widely does the optimal threshold vary?
  - at the single GLOBAL-EER threshold, the per-speaker FRR/FAR spread (how many
    speakers are badly served by one threshold)
  - adaptive gain: mean per-speaker EER vs global EER
  - risk-coverage (abstention): abstain on the lowest-|margin| queries; error
    among retained vs coverage

Outputs output/analysis/threshold_analysis.json and output/fig_threshold.png.
Usage: python -m src.threshold_analysis --models ecapa animeva
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AN = Path("output/analysis")

DISP_NAMES = {
    "ecapa": "ECAPA-TDNN", "animeva": "animeva", "campp": "CAM++",
    "redimnet": "ReDimNet-b2", "xvector": "x-vector", "wavlm": "WavLM-base-plus-sv",
    "jhubert": "JP-HuBERT", "hubert": "JP-HuBERT", "jxvector": "jxvector",
    "ens_sv4": "SV-4", "ens_all6": "all-6", "seedvc": "Seed-VC",
    "irodori": "Irodori-TTS", "gptsovits": "GPT-SoVITS", "gptsovits_v1": "GPT-SoVITS v1",
    "gptsovits_v2": "GPT-SoVITS v2", "gptsovits_v2ProPlus": "GPT-SoVITS v2ProPlus",
    "gptsovits_v3": "GPT-SoVITS v3", "gptsovits_v4": "GPT-SoVITS v4", "real": "real",
}


def disp(k):
    return DISP_NAMES.get(str(k), str(k))


def eer_threshold(gen, imp):
    s = np.concatenate([gen, imp]); y = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); y = y[o]; s = s[o]
    P, N = y.sum(), len(y) - y.sum()
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    fnr = 1 - tp / P; fpr = fp / N
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float(s[i]), float((fnr[i] + fpr[i]) / 2)


def run(model):
    d = np.load(AN / f"{model}.npz", allow_pickle=True)
    sb = d["same_best"].astype(np.float32); db = d["diff_best"].astype(np.float32)
    spk = np.asarray(d["speaker_id"]); has = d["has_same"]
    vm = d["verification_margin"].astype(np.float32)
    gen, imp = sb[has], db[has]
    tau_g, eer_g = eer_threshold(gen, imp)

    by = defaultdict(list)
    for i in np.where(has)[0]:
        by[spk[i]].append(i)
    taus, eers, frr_g, far_g = [], [], [], []
    for s, idx in by.items():
        idx = np.array(idx)
        g, im = sb[idx], db[idx]
        if len(g) < 4:
            continue
        t, e = eer_threshold(g, im); taus.append(t); eers.append(e)
        frr_g.append(float((g < tau_g).mean())); far_g.append(float((im > tau_g).mean()))
    taus = np.array(taus); eers = np.array(eers); frr_g = np.array(frr_g); far_g = np.array(far_g)

    # risk-coverage via abstaining on lowest-confidence (|margin|) queries
    conf = np.abs(vm[has]); mis = (vm[has] < 0).astype(float)
    order = np.argsort(-conf)  # keep most-confident first
    mis_sorted = mis[order]
    cov = np.arange(1, len(mis_sorted) + 1) / len(mis_sorted)
    err = np.cumsum(mis_sorted) / np.arange(1, len(mis_sorted) + 1)
    def err_at(c):
        k = max(1, int(c * len(mis_sorted)))
        return float(mis_sorted[:k].mean())

    res = {
        "model": model, "global_eer": eer_g, "global_threshold": tau_g,
        "per_speaker_threshold_std": float(taus.std()),
        "per_speaker_threshold_iqr": float(np.percentile(taus, 75) - np.percentile(taus, 25)),
        "per_speaker_threshold_range_5_95": [float(np.percentile(taus, 5)), float(np.percentile(taus, 95))],
        "mean_per_speaker_eer": float(eers.mean()),
        "adaptive_gain_eer": float(eer_g - eers.mean()),
        "at_global_thr_frac_speakers_FRR>20%": float((frr_g > 0.2).mean()),
        "at_global_thr_frac_speakers_FAR>20%": float((far_g > 0.2).mean()),
        "abstain_error_at_coverage": {f"{c:.0%}": err_at(c) for c in [1.0, 0.9, 0.75, 0.5]},
    }
    print(f"\n#### {model} ####")
    print(f"  global EER={eer_g*100:.1f}% | mean per-speaker EER={eers.mean()*100:.1f}% "
          f"(adaptive gain {res['adaptive_gain_eer']*100:.1f}pp)")
    print(f"  per-speaker EER-threshold: std={taus.std():.3f} 5-95%={res['per_speaker_threshold_range_5_95'][0]:.2f}..{res['per_speaker_threshold_range_5_95'][1]:.2f}")
    print(f"  at single global threshold: {res['at_global_thr_frac_speakers_FRR>20%']*100:.0f}% of speakers have FRR>20%, "
          f"{res['at_global_thr_frac_speakers_FAR>20%']*100:.0f}% have FAR>20%")
    print(f"  abstain risk-coverage misID: " + "  ".join(f"{k}:{v*100:.1f}%" for k, v in res["abstain_error_at_coverage"].items()))
    return res, taus, tau_g, (cov, err)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["ecapa", "animeva"])
    args = ap.parse_args()
    out = {}; plotdata = {}
    for m in args.models:
        res, taus, tau_g, rc = run(m); out[m] = res; plotdata[m] = (taus, tau_g, rc)
    # figure: per-speaker threshold spread + risk-coverage
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, (m, (taus, tau_g, (cov, err))) in enumerate(plotdata.items()):
        c = f"C{i}"  # hist and its global-thr line share a color (C0 blue ecapa, C1 orange animeva)
        ax[0].hist(taus, bins=40, alpha=0.5, color=c, label=f"{disp(m)} (per-speaker EER thr)")
        ax[0].axvline(tau_g, ls="--", lw=1.5, color=c, label=f"{disp(m)} global thr")
        ax[1].plot(cov * 100, err * 100, lw=2, color=c, label=disp(m))
    ax[0].set_xlabel("cosine threshold"); ax[0].set_ylabel("# speakers")
    # legends below each panel (above the caption), outside the axes so they never overlap the data
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)
    ax[1].set_xlabel("coverage (% retained)"); ax[1].set_ylabel("misID among retained (%)")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)
    fig.tight_layout(); fig.savefig("output/fig_threshold.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("\nwrote output/fig_threshold.png")
    AN.mkdir(parents=True, exist_ok=True); json.dump(out, open(AN / "threshold_analysis.json", "w"), indent=2)


if __name__ == "__main__":
    main()

"""Paper-grade figures:
  output/fig_style_controlled.png  -- VA vs controls (JVS-parallel/varied, CV), EER
  output/fig_clone_detection.png   -- clone detectability & false-attribution
                                       (open 795-actor gallery, with bootstrap CIs;
                                        kNN-VC closed-gallery shown as the collapse)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = Path("output/analysis")


def fig_style_controlled():
    d = json.load(open(A / "pop_table.json"))
    models = list(d)
    ctrl_order = [("jvs", "JVS\n(neutral)"), ("jvsv", "JVS\n(varied)"), ("cv", "CommonVoice\n(general)")]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        labels, va_e, ct_e = [], [], []
        for key, lab in ctrl_order:
            if key in d[m]:
                labels.append(lab)
                va_e.append(d[m][key]["VA"]["eer"] * 100)
                ct_e.append(d[m][key][key]["eer"] * 100)
        x = np.arange(len(labels)); w = 0.38
        ax.bar(x - w / 2, va_e, w, label="Voice actors", color="#c0392b")
        ax.bar(x + w / 2, ct_e, w, label="Control", color="#2980b9")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(m, fontsize=11); ax.set_ylabel("EER (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Voice actors vs control populations (AS-norm, matched size) — "
                 "gap shrinks as the control gains intra-speaker style range")
    fig.tight_layout()
    fig.savefig("output/fig_style_controlled.png", dpi=150)
    plt.close(fig)
    print("wrote output/fig_style_controlled.png")


def fig_clone_detection():
    sets = []
    # Seed-VC (zero-shot SOTA VC) replaces the low-fidelity kNN-VC control;
    # Seed-VC and Irodori scored at 120 targets, GPT-SoVITS at 40.
    for label, fname in [("seedvc", "seedvc"), ("irodori", "irodori120"), ("gptsovits", "gptsovits")]:
        p = A / f"clone_score_{fname}.json"
        if not p.exists():
            p = A / f"clone_score_{label}.json"
        if p.exists():
            sets.append((label, json.load(open(p))))
    closed = {}

    models = ["animeva", "ecapa"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics = [("detectable", "Detection rate @ EER (%)"), ("false_attr", "False attribution (%)")]
    ci_key = {"detectable": "det", "false_attr": "false"}
    for ax, (metric, ylab) in zip(axes, metrics):
        groups = [f"{lbl}\n{m}" for lbl, _ in sets for m in models]
        vals, errs, colors = [], [], []
        for lbl, res in sets:
            for m in models:
                r = res[m]
                v = r[metric] * 100
                ci = r["ci95"][ci_key[metric]]
                vals.append(v)
                errs.append([[v - ci[0] * 100], [ci[1] * 100 - v]])
                colors.append("#27ae60" if m == "animeva" else "#e67e22")
        x = np.arange(len(groups))
        for i in range(len(groups)):
            ax.bar(x[i], vals[i], 0.7, color=colors[i],
                   yerr=np.array(errs[i]), capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylabel(ylab); ax.set_ylim(0, 100)
        # reference: kNN-VC closed-gallery (optimistic) for this metric
        if closed:
            ck = "detectable_rate" if metric == "detectable" else "false_attribution_rate"
            for m, c in [("animeva", "#27ae60"), ("ecapa", "#e67e22")]:
                if m in closed:
                    ax.axhline(closed[m][ck] * 100, ls="--", lw=1, color=c, alpha=0.6)
    axes[0].plot([], [], color="#27ae60", label="animeva (JA VA-domain)")
    axes[0].plot([], [], color="#e67e22", label="ECAPA (EN, off-the-shelf)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Unauthorized-clone detection on a realistic 795-actor gallery "
                 "(open set) — fixed threshold misses many clones and misattributes")
    fig.tight_layout()
    fig.savefig("output/fig_clone_detection.png", dpi=150)
    plt.close(fig)
    print("wrote output/fig_clone_detection.png")


def fig_version_gradient():
    """GPT-SoVITS family: clone fidelity (version) vs detectability/false-attribution."""
    order = [("gptsovits_v1", "v1\n(2024)"), ("gptsovits_v2", "v2"),
             ("gptsovits", "v2ProPlus"), ("gptsovits_v3", "v3"), ("gptsovits_v4", "v4\n(latest)")]
    data = {}
    for key, lab in order:
        p = A / f"clone_score_{key}.json"
        if p.exists():
            data[lab] = json.load(open(p))
    labs = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, ylab in [(axes[0], "detectable", "Detection rate @ EER (%)"),
                             (axes[1], "false_attr", "False attribution (%)")]:
        for model, c in [("animeva", "#27ae60"), ("ecapa", "#e67e22")]:
            ys = [data[l][model][metric] * 100 for l in labs]
            ci = {"detectable": "det", "false_attr": "false"}[metric]
            lo = [data[l][model][metric] * 100 - data[l][model]["ci95"][ci][0] * 100 for l in labs]
            hi = [data[l][model]["ci95"][ci][1] * 100 - data[l][model][metric] * 100 for l in labs]
            ax.errorbar(range(len(labs)), ys, yerr=[lo, hi], marker="o", capsize=3,
                        color=c, label=("animeva (JA)" if model == "animeva" else "ECAPA (EN)"))
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab); ax.set_ylim(0, 100); ax.legend(fontsize=8)
    fig.suptitle("GPT-SoVITS clone fidelity vs detector behaviour: higher-fidelity (newer) "
                 "clones are MORE detectable; low-fidelity clones cause wrongful attribution")
    fig.tight_layout(); fig.savefig("output/fig_version_gradient.png", dpi=150); plt.close(fig)
    print("wrote output/fig_version_gradient.png")


if __name__ == "__main__":
    fig_style_controlled()
    fig_clone_detection()
    fig_version_gradient()

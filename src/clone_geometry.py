"""Story A: does a clone's embedding cluster by SYNTHESIS METHOD or by TARGET
speaker? If by method, threshold "unauthorized-generation detection" is really
anti-spoofing (synthetic-vs-real), not speaker attribution.

Embeds every clone from every method + some real target audio, then measures:
  - nearest-neighbour: is a clone's nearest other clone same-method vs same-target
    (each vs its chance level -> enrichment)
  - linear-probe accuracy: predict METHOD vs predict TARGET from the embedding
    (each vs its chance level)
  - same-target/diff-method cosine vs same-method/diff-target cosine
  - real-vs-synthetic separability (binary linear probe accuracy)
Also writes a 2-D UMAP colored by method and by target.

Usage: python -m src.clone_geometry --models animeva ecapa
"""
from __future__ import annotations

import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch, soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score

from src.clone_probe import va_actor_segments
from src.embeddings import build_extractor

CLONES = {
    # kNN-VC excluded: perceptual fidelity too low to be a realistic attack (its
    # unit-stitching artifacts also confound a "channel" interpretation). Replaced by
    # Seed-VC v2 (Japanese-capable zero-shot SOTA voice conversion).
    "seedvc": "/path/to/va-data/clones/seedvc",
    "irodori": "/path/to/va-data/clones/irodori",
    "gptsovits_v1": "/path/to/va-data/clones/gptsovits_v1",
    "gptsovits_v2": "/path/to/va-data/clones/gptsovits_v2",
    "gptsovits_v2ProPlus": "/path/to/va-data/clones/gptsovits",
    "gptsovits_v3": "/path/to/va-data/clones/gptsovits_v3",
    "gptsovits_v4": "/path/to/va-data/clones/gptsovits_v4",
}

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


def embed_dir(ext, d, method, want_targets):
    rows = []
    for f in sorted(Path(d).glob("clone_*__src*.wav")):
        m = re.match(r"clone_(.+)__src\d+", f.stem)
        if not m or m.group(1) not in want_targets:
            continue
        w, sr = sf.read(str(f), dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
        assert sr == 16000, f"{f}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
        v = np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1)
        rows.append((v, method, m.group(1)))
    return rows


def run(model):
    ext = build_extractor(model, "cuda")
    actors = va_actor_segments()
    want = set(actors.keys())
    rows = []
    for method, d in CLONES.items():
        if Path(d).exists():
            rows += embed_dir(ext, d, method, want)
    # real audio for the same targets (held-out-ish: segments 8..12)
    real_targets = {t for _, _, t in rows}
    for t in real_targets:
        for p in actors[t][8:12]:
            w, sr = sf.read(p, dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
            assert sr == 16000, f"{p}: expected 16 kHz, got {sr} Hz (embed() does not resample)"
            rows.append((np.asarray(ext.embed(torch.from_numpy(w)), np.float32).reshape(-1), "real", t))

    X = np.stack([r[0] for r in rows]); method = np.array([r[1] for r in rows]); target = np.array([r[2] for r in rows])
    X = X - X.mean(0, keepdims=True); X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    syn = method != "real"
    Xs, ms, ts = X[syn], method[syn], target[syn]   # synthetic only for method/target clustering

    # nearest-neighbour enrichment (synthetic clones only)
    S = Xs @ Xs.T; np.fill_diagonal(S, -9)
    nn = S.argmax(1)
    same_method = float(np.mean(ms[nn] == ms)); same_target = float(np.mean(ts[nn] == ts))
    cm = Counter(ms.tolist()); ct = Counter(ts.tolist()); n = len(ms)
    chance_method = float(np.mean([(cm[x] - 1) / (n - 1) for x in ms]))
    chance_target = float(np.mean([(ct[x] - 1) / (n - 1) for x in ts]))

    # linear probes (vs chance)
    def probe(y):
        acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), Xs, y, cv=4).mean()
        chance = max(Counter(y).values()) / len(y)
        return float(acc), float(chance)
    acc_m, ch_m = probe(ms); acc_t, ch_t = probe(ts)

    # same-target/diff-method vs same-method/diff-target cosine
    st_dm, sm_dt = [], []
    for i in range(0, n, 3):  # subsample rows for speed
        same_t = ts == ts[i]; same_m = ms == ms[i]
        a = S[i][same_t & ~same_m]; b = S[i][same_m & ~same_t]
        if len(a): st_dm.append(a.mean())
        if len(b): sm_dt.append(b.mean())

    # real-vs-synthetic separability
    yb = (~syn).astype(int)  # 1=real
    acc_rs = cross_val_score(LogisticRegression(max_iter=2000), X, yb, cv=4).mean()

    sil_method = silhouette_score(Xs, ms, metric="cosine", sample_size=min(2000, n), random_state=0)
    sil_target = silhouette_score(Xs, ts, metric="cosine", sample_size=min(2000, n), random_state=0)

    res = {
        "model": model, "n_synth": int(n), "n_methods": len(set(ms)), "n_targets": len(set(ts)),
        "nn_same_method": same_method, "nn_chance_method": chance_method,
        "nn_same_target": same_target, "nn_chance_target": chance_target,
        "probe_method_acc": acc_m, "probe_method_chance": ch_m,
        "probe_target_acc": acc_t, "probe_target_chance": ch_t,
        "cos_same_target_diff_method": float(np.mean(st_dm)),
        "cos_same_method_diff_target": float(np.mean(sm_dt)),
        "silhouette_by_method": float(sil_method), "silhouette_by_target": float(sil_target),
        "real_vs_synth_probe_acc": float(acc_rs),
    }
    print(f"\n#### {model} (synthetic clones n={n}, {res['n_methods']} methods, {res['n_targets']} targets) ####")
    print(f"  NN same-METHOD: {same_method*100:.1f}% (chance {chance_method*100:.1f}) -> {same_method/chance_method:.1f}x")
    print(f"  NN same-TARGET: {same_target*100:.1f}% (chance {chance_target*100:.1f}) -> {same_target/chance_target:.1f}x")
    print(f"  probe METHOD acc: {acc_m*100:.1f}% (chance {ch_m*100:.1f})")
    print(f"  probe TARGET acc: {acc_t*100:.1f}% (chance {ch_t*100:.1f})")
    print(f"  cos same-target/diff-method={res['cos_same_target_diff_method']:.3f}  vs  same-method/diff-target={res['cos_same_method_diff_target']:.3f}")
    print(f"  silhouette by METHOD={sil_method:.3f}  by TARGET={sil_target:.3f}")
    print(f"  real-vs-synthetic probe acc: {acc_rs*100:.1f}%")
    return res, X, method, target


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["animeva", "ecapa"])
    ap.add_argument("--umap", action="store_true"); args = ap.parse_args()
    out = {}
    for m in args.models:
        res, X, method, target = run(m)
        out[m] = res
        if args.umap:
            try:
                import umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                plt.rcParams.update({
                    "font.size": 13, "axes.labelsize": 13,
                    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
                })
                xy = umap.UMAP(n_components=2, metric="cosine", random_state=0).fit_transform(X)
                fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
                for a, lab in [(ax[0], method), (ax[1], target)]:
                    uniq = sorted(set(lab)); cmap = plt.cm.tab20(np.linspace(0, 1, len(uniq)))
                    for c, u in zip(cmap, uniq):
                        sel = lab == u; a.scatter(xy[sel, 0], xy[sel, 1], s=10, color=c, alpha=0.7,
                                                  label=disp(u) if len(uniq) <= 10 else None)
                panels = [(ax[0], "(a) colored by synthesizer"),
                          (ax[1], "(b) colored by target speaker (120 targets, no legend)")]
                for a, panel in panels:
                    a.set_xticks([]); a.set_yticks([])
                    a.annotate(panel, xy=(0.5, 1.02), xycoords="axes fraction",
                               ha="center", va="bottom", fontsize=12, fontweight="bold")
                fig.tight_layout()
                # synthesizer legend anchored UNDER THE LEFT PANEL ONLY (it keys panel (a);
                # the right panel is coloured by target speaker and has no legend), with an
                # explicit title so it is never mistaken for a figure-wide legend
                if len(set(method)) <= 10:
                    h, l = ax[0].get_legend_handles_labels()
                    leg = ax[0].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                                       ncol=3, markerscale=2, frameon=True,
                                       title="synthesizer — applies to panel (a) only")
                    leg.get_title().set_fontweight("bold")
                fig.savefig(f"output/fig_clone_geometry_{m}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
                print(f"  wrote output/fig_clone_geometry_{m}.png")
            except Exception as e:
                print(f"  umap skip: {e}")
    Path("output/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("output/analysis/clone_geometry.json", "w"), indent=2)


if __name__ == "__main__":
    main()

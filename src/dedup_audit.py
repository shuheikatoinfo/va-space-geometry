"""Speaker de-duplication / homonym audit (reviewer: cross-source duplicate speakers
or homonym merges could manufacture the thin-margin / high-density signal).

Two independent checks:
  (1) Registry homonyms: how many normalized speaker NAMES map to >1 registry id
      (a name collision would, if merged, inject spurious intra-speaker variance;
      here they are kept as distinct ids, so we just count the exposure).
  (2) Embedding near-duplicate speakers: cross-id centroid cosine above a
      genuine-level threshold flags a candidate SPLIT identity (one real person
      under two ids -> a same-person pair scored as an impostor near-tie). This is
      an UPPER BOUND (very similar distinct voices also land here); we report the
      count at several thresholds and re-estimate the misID floor with the flagged
      pairs' higher-id members removed.

Usage: python -m src.dedup_audit --model animeva
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np

EMB = Path("output/embeddings"); AN = Path("output/analysis")


def norm_name(s):
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").replace("　", "").strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="animeva"); args = ap.parse_args()
    # (1) registry homonyms
    name2ids = defaultdict(set)
    for l in open("data/registry/speakers.jsonl", encoding="utf-8"):
        r = json.loads(l); name2ids[norm_name(r.get("name", ""))].add(str(r["speaker_id"]))
    homonyms = {n: sorted(ids) for n, ids in name2ids.items() if n and len(ids) > 1}
    n_names = len([n for n in name2ids if n])
    print(f"Registry: {n_names} distinct normalized names; "
          f"{len(homonyms)} names map to >1 id ({sum(len(v) for v in homonyms.values())} ids, "
          f"{len(homonyms)/n_names*100:.2f}% of names).")

    # (2) embedding near-duplicate speakers via centroid cosine
    d = np.load(EMB / f"{args.model}.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32); spk = np.asarray(d["speaker_id"]).astype(str)
    src = np.asarray(d["recording_source"])
    keep = np.array([i for i in range(len(spk)) if not str(src[i]).startswith("freelance:")])
    emb, spk = emb[keep], spk[keep]
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    ids = sorted(set(spk.tolist()))
    C = np.stack([emb[spk == s].mean(0) for s in ids])
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)
    S = C @ C.T; np.fill_diagonal(S, -9.0)
    iu = np.triu_indices(len(ids), 1)
    pair_cos = S[iu]
    print(f"\n{args.model} centroids: {len(ids)} speakers, {len(pair_cos)} cross-speaker pairs")
    print(f"  impostor centroid cosine: mean {pair_cos.mean():.3f}  p99 {np.percentile(pair_cos,99):.3f}  "
          f"p99.9 {np.percentile(pair_cos,99.9):.3f}  max {pair_cos.max():.3f}")
    flagged = {}
    ii, jj = np.triu_indices(len(ids), 1)
    for thr in [0.70, 0.80, 0.90]:
        cnt = int(np.sum(pair_cos > thr))
        flagged[thr] = cnt
        print(f"  pairs with centroid cosine > {thr:.2f}: {cnt}  "
              f"({cnt/len(ids)*100:.2f}% of #speakers = candidate split-id upper bound)")
    # leave-flagged-pairs-out floor re-estimate: drop the higher-id member of every
    # candidate-duplicate pair (centroid cosine > 0.70) and recompute the raw-cosine
    # misID floor on the agency subset. If the floor is unchanged, the residual is
    # not manufactured by cross-source duplicate/split identities.
    from src.verification_metrics import openset_misid, ln
    Xn = ln(emb)
    misid_full = openset_misid(Xn, spk)
    sel = pair_cos > 0.70
    drop_speakers = set()
    for a, b in zip(ii[sel], jj[sel]):
        drop_speakers.add(ids[max(a, b)])  # drop the higher registry-id member
    keep_mask = np.array([s not in drop_speakers for s in spk])
    misid_dedup = openset_misid(Xn[keep_mask], spk[keep_mask])
    print(f"  raw-cosine misID floor: full {misid_full*100:.2f}%  ->  "
          f"after dropping {len(drop_speakers)} candidate-duplicate speaker(s): {misid_dedup*100:.2f}%")
    out = {"model": args.model, "n_names": n_names, "n_homonym_names": len(homonyms),
           "homonym_examples": dict(list(homonyms.items())[:10]),
           "n_speakers": len(ids), "impostor_centroid_cos_mean": float(pair_cos.mean()),
           "impostor_centroid_cos_p999": float(np.percentile(pair_cos, 99.9)),
           "impostor_centroid_cos_max": float(pair_cos.max()),
           "flagged_pairs": {str(k): v for k, v in flagged.items()},
           "n_dropped_speakers": len(drop_speakers),
           "misID_floor_full": float(misid_full),
           "misID_floor_dedup": float(misid_dedup)}
    AN.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(AN / "dedup_audit.json", "w"), ensure_ascii=False, indent=2)
    print("\n-> output/analysis/dedup_audit.json")


if __name__ == "__main__":
    main()

# va-space-geometry — Stage 1 analysis code

Analysis code for the paper:

> **A Geometry-Limited Identification Floor and Its Consequences for
> Voice-Clone Attribution in Professional Voice Actors**
> Shuhei Kato. (arXiv preprint / IEEE Access submission; citation to be added
> at preprint v1.)

Canonical repository: <https://github.com/shuheikatoinfo/va-space-geometry>

A speaker-embedding space may *look* correct (the right speaker is the nearest
neighbor) while leaving almost no decision margin. On 1,168 professional
Japanese voice actors, the paper shows a **misidentification floor that
survives calibration, score normalization, and discriminative re-ranking**
(LDA / WCCN / two-covariance PLDA — moment-fit and EM alike), and quantifies
the two coupled harms this produces for similarity-threshold voice-clone
attribution: **missed attribution** and **wrongful accusation** at one
operating point.

## What this repository contains (release Stage 1)

The **analysis pipeline**: gallery construction, hubness / rank-1 misID / EER
computation, the back-end suite (LDA / WCCN / PLDA, EM refit, nonlinear
re-rankers), the clone probes and confound controls, and the figure generators.

The **core geometry and back-end suite are data-independent** and run on **any
embedding set conforming to the `.npz` format below**, without the voice-actor
corpus. The **clone probes, demographic and session-disjoint analyses, and
audio-based controls additionally require withheld inputs**: the segment-level
metadata (`data/processed/segments.jsonl` — segment paths and `source_sha256`),
per-speaker registry fields (`data/registry/speakers.jsonl` — e.g. gender,
credits), and, for the clone and codec/BGM controls, the corresponding audio.
Those inputs are part of the withheld tier (below), available to
reviewers/replicators on request.

Consistent with the paper's ethics section, this repository ships **no audio,
no scrapers, and no per-speaker artifacts**:

- **Data-collection modules (`src/scrape/`) are not distributed.** Two
  preprocessing entry points (`src/extract.py`, `src/build_dataset.py`) carry a
  local timestamp shim in place of the collection helper they imported.
- **Stage 2 — anonymized derived artifacts — are in [`artifacts/`](artifacts/)**
  (the builder is intentionally withheld: it holds the secret salt and the
  id↔hash mapping): per-encoder derived statistics,
  speaker-disjoint split definitions (seeds 0–2), and the animeva
  training-overlap hash set, all keyed by **salted one-way hashes** of the
  registry id (HMAC-SHA256, salt withheld; id↔hash mapping available to
  reviewers/replicators only under the data-use agreement). Sanitization
  rules are documented in [`artifacts/MANIFEST.md`](artifacts/MANIFEST.md).
- **Raw embeddings are withheld — not publicly released.** Per-segment
  embeddings, per-speaker centroids, and paired trial lists are **biometric
  identifiers**. **They are pseudonymous, not anonymous:** hash-keying
  anonymizes only the identifier column, while the vectors themselves are
  identity-linked and — because they were produced by *public* encoders —
  re-identifiable by nearest-neighbour matching against public audio, **with
  no salt required** (individual-identification codes under Japan's APPI).
  Consistent with the paper's own findings on this re-identifiability, they
  are **not** published. They are available to reviewers/replicators **on
  request**, under a data-use agreement (research use only, no
  re-identification, no redistribution, verification only). The full pipeline
  reproduces on any embedding set in the format below, including the consented
  control corpora used in the paper (JVS, CommonVoice).
- **Raw collected audio and generated clones are never redistributed.**

## Embedding `.npz` format

Each `output/embeddings/<model>.npz` holds row-aligned arrays:

| key | contents |
|---|---|
| `emb` | `float32 (n_segments, dim)` per-segment embeddings |
| `speaker_id` | length-`n_segments` speaker id per segment |
| `segment_id` | per-segment identifier |
| `style_label` | style bucket (narration / dialogue / …) |
| `recording_source` | channel/source label (used by confound controls) |

Any encoder's embeddings drop in under this layout.

## Environment

- Python 3.10 (tested on CPython 3.10.18); `pip install -r requirements.txt`
  (**pinned versions** — the exact stack used to produce the reported numbers).
- Embedding extraction uses a CUDA GPU; the analysis stage runs CPU-only on
  precomputed embeddings (the 1:N misID scorers use CUDA when available).

## Reproducing the analysis

Stage 1 modules read `output/embeddings/<model>.npz` and write metrics to
`output/analysis/` and figures to `output/`:

```bash
# Verification back-ends: EER, minDCF, Cllr, 1:N closed-set misID
# (cosine -> LDA -> WCCN -> PLDA re-ranking suite)
python -m src.verification_plda --models ens_sv4 animeva ecapa --n-splits 3
python -m src.verification_metrics
python -m src.em_plda                 # EM/ML two-covariance PLDA estimator control

# Raw geometry: rank-1 identification, decision margins, hubness
python -m src.analyze
python -m src.plot

# Per-speaker thresholds and confusion-graph structure
python -m src.threshold_analysis
python -m src.confusion_graph
python -m src.confusion_null
python -m src.fig_confusion

# Defensive clone probe and clone geometry
python -m src.centered_gallery
python -m src.score_clones
python -m src.clone_geometry
python -m src.source_enrichment

# Session-disjoint splits, learned fusion, fairness
python -m src.session_disjoint_backend
python -m src.calibrated_eval --session-disjoint  # writes session_disjoint.json (Table-2 metrics under a source-disjoint genuine constraint)
python -m src.learned_fusion
python -m src.fairness_analysis
```

## Model and system versions

Exact source IDs and revisions of every speaker encoder and clone/TTS system used
to produce the reported results (recorded from the working trees and model
caches). Python dependency versions are pinned in [`requirements.txt`](requirements.txt).

**Speaker encoders**

| Encoder | Source | Revision / checkpoint |
|---|---|---|
| x-vector | HF `speechbrain/spkrec-xvect-voxceleb` | `56895a2df401be4150a159f3a1c653f00051d477` |
| ECAPA-TDNN | HF `speechbrain/spkrec-ecapa-voxceleb` | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |
| WavLM-base-plus-sv | HF `microsoft/wavlm-base-plus-sv` | `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` |
| CAM++ | ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` | `v2.0.2` |
| ReDimNet-b2 | torch.hub `IDRnD/ReDimNet` (`model_name=b2, train_type=ptn`) | ckpt `b2-vox2-ptn.pt` |
| JP-HuBERT (layer-mean) | HF `reazon-research/japanese-hubert-base-k2` | `a9f26026165f8b80256f0aeecee53dedf81abce1` |
| jxvector | torch.hub `sarulab-speech/xvector_jtubespeech` | ckpt `xvector.pth` |
| animeva | `litagin/anime_speaker_embedding` (pip package `anime_speaker_embedding`) | `0.2.1` (pinned in [`requirements.txt`](requirements.txt)) |

Ensembles are embedding-level fusions of the above: **SV-4** = {x-vector,
ECAPA-TDNN, CAM++, ReDimNet-b2}; **all-6** = {x-vector, ECAPA-TDNN, WavLM, CAM++,
ReDimNet-b2, JP-HuBERT}. The torch.hub encoders load from the repository default
branch; the checkpoint filename pins the weights.

**Clone / TTS systems and copy-synthesis control**

| System | Source | Revision |
|---|---|---|
| GPT-SoVITS (v1–v4, v2ProPlus runs) | github.com/RVC-Boss/GPT-SoVITS | commit `bf81cdb14a38b674b6e9996dabc97340bc9978d2` (2026-06-20) |
| Seed-VC v2 | github.com/Plachtaa/seed-vc | commit `51383efd921027683c89e5348211d93ff12ac2a8` (2025-04-20); default v2 AR/CFM checkpoints, 30 diffusion steps |
| Irodori-TTS (inference repo) | github.com/Aratako/Irodori-TTS | commit `eaf74d6a19138f743acb5b71a445fd25a57db987` (2026-06-04) |
| Irodori-TTS-500M-v3 | HF `Aratako/Irodori-TTS-500M-v3` | `236c1e56591279fc24e3c1bf6609fc06e48dde28` |
| Irodori-TTS-600M-v3-VoiceDesign | HF `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | `e863a3a93e652e09afeff3e84823a206a0a60314` |
| Semantic-DACVAE-Japanese-32dim (Irodori codec) | HF `Aratako/Semantic-DACVAE-Japanese-32dim` | `47376ee24834d7a05a48ebabfe3cde29b3c5e214` |
| BigVGAN v2 22 kHz 80-band (copy-synthesis control) | HF `nvidia/bigvgan_v2_22khz_80band_256x` | `633ff708ed5b74903e86ff1298cf4a98e921c513` |

GPT-SoVITS pretrained weight pairs per version — v1: `s1bert25hz-2kh-…ckpt` +
`s2G488k.pth`; v2: `…5kh-…ckpt` + `s2G2333k.pth`; v3: `s1v3.ckpt` + `s2Gv3.pth`;
v4: `s1v3.ckpt` + `s2Gv4.pth`; v2ProPlus: `s1v3.ckpt` + `s2Gv2ProPlus.pth`.

## Archival and citation

A citable snapshot of this repository is archived on Zenodo with a permanent DOI:

> **Concept DOI (all versions):** [`10.5281/zenodo.21368121`](https://doi.org/10.5281/zenodo.21368121)
> — always resolves to the latest archived version; cite this for the software in general.
> **This version (v1.0.0):** [`10.5281/zenodo.21368122`](https://doi.org/10.5281/zenodo.21368122).
> Please cite both the paper and this record (see [`CITATION.cff`](CITATION.cff)).

Zenodo snapshots are **immutable**: a published version cannot be altered after
the fact. Corrections and removals are applied to the live repository and to all
subsequent Zenodo versions (under the same concept DOI); earlier immutable
snapshots remain as archived.

## Removal and objection

Any voice actor represented in these derived artifacts — or their agency — may
**request removal or exclusion** of their hash-keyed artifacts, or object to
their inclusion, by contacting the author (below). Removals are honored at the
repository HEAD, in every future released version, and in the withheld,
on-request embedding / mapping tier. Because archived Zenodo snapshots are
immutable (above), a removal cannot rewrite a prior snapshot, but it is applied
to the live repository and to all subsequent archived versions.

## License

Three tiers, each with its own terms:

- **Code** (`src/`) — [MIT](LICENSE).
- **Derived artifacts** (`artifacts/`) — [CC BY 4.0](artifacts/LICENSE)
  (attribution: Shuhei Kato, *va-space-geometry*). These are anonymized,
  hash-keyed aggregate statistics.
- **Raw embeddings** (withheld, on request) — biometric data, not publicly
  released; available to reviewers/replicators on request under a bespoke
  Data-Use Agreement (research use only, no re-identification, no
  redistribution, verification only).

No license grants any rights over the underlying audio or the actors' voices.

## Contact

Shuhei Kato — shuhei@shuheikato.info (data-use agreement requests: same
address; research use only, no re-identification, no redistribution).

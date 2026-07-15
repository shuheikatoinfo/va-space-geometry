# Stage 2 derived artifacts (anonymized)

All per-speaker keys are **salted one-way hashes** of the registry ID
(HMAC-SHA256, first 16 hex characters; salt withheld by the author). The
ID<->hash mapping is available to reviewers/replicators only under the
data-use agreement described in the paper (research use only, no
re-identification, no redistribution).

The salt is the single load-bearing secret: the registry-ID space is small
(~1,168 IDs), so **disclosure of the salt makes every hash trivially
reversible** and re-identifies all actors, and **loss of the salt is
unrecoverable** (future releases could not reproduce these keys). It is kept
out of every repository and every artifact.

- `analysis/` — per-encoder derived statistics. Sanitization applied:
  per-clone `rows` dropped (aggregate rates kept, as pledged: clone-probe
  outputs would otherwise name innocent wrongfully-attributed actors);
  hub/absorber example name lists dropped; homonym examples dropped;
  agency keys hashed; file lists with local paths dropped; confusable-pair
  IDs and the jvs001 nearest-actor ID hashed.
- `splits/` — speaker-disjoint 55/45 split definitions (seeds 0-2),
  hash-keyed; these are the splits behind the back-end tables.
- `animeva_train_overlap_hashes.json` — hash -> bool map enabling the
  training-overlap audit without revealing identities.

Embeddings (per-segment and per-speaker centroids) are **biometric
identifiers** and are **not publicly released**: they are available to
reviewers/replicators on request under a data-use agreement (see README), for
verification only. They are **pseudonymous, not anonymous** — hash-keying
anonymizes only the identifier column, while the vectors, produced by public
encoders, remain re-identifiable by nearest-neighbour matching without the
salt.

# Retrieval V2 — scalable sparse candidate generation

## Status

This branch restarts candidate-retrieval optimization from the validated Stage 4
evidence. It does not discard the validated normalization, preprocessing,
ground-truth evaluation, or hybrid name/address signals.

The previous `NearestNeighbors` implementation remains the correctness
baseline only.

## Evidence that motivates the restart

Measured with 1,000 India queries, chunk size 250, name K=20, address K=10 and
final K=30:

| Total targets | Peak RSS | Total runtime |
|---:|---:|---:|
| 125,000 | 1.64 GB | 20.27 s |
| 250,000 | 2.54 GB | 36.41 s |
| 500,000 | 4.14 GB | 64.04 s |

Candidate bounds remained correct. The problem is therefore not unbounded
output; it is the cost of discovering Top-K candidates.

scikit-learn documents that fitting NearestNeighbors on sparse input forces
brute-force search. That makes the current implementation unsuitable as the
production retrieval engine at multi-million-target scale.

## Design decision

Preserve the validated sparse character n-gram retrieval signal and change the
search algorithm before changing the representation.

Production direction:

1. exact/high-precision blocks remain a cheap retrieval channel;
2. sparse character n-gram fuzzy retrieval supplies bounded Top-K candidates;
3. name and address channels are evaluated independently;
4. candidate IDs are unioned/deduplicated and hard-capped;
5. downstream pair features and LightGBM are built only after retrieval passes
   recall, RAM and runtime gates.

## Algorithm ladder

### A. Exact sparse Top-K multiplication — first experiment

Benchmark a Top-N sparse matrix multiplication implementation against the
existing brute-force baseline on the *same* TF-IDF representation.

Why first:
- preserves the retrieval signal already validated;
- directly computes the operation we need: bounded high-similarity sparse pairs;
- gives an exact-result/equivalence experiment before approximate methods.

No dependency is added until a compatibility/licence/install check passes.

### B. Blockwise sparse Top-K — bounded-memory fallback/reference

Process target blocks independently and merge a running per-query Top-K.
Target block size, not total target count, bounds the resident target matrix.

This can solve RAM growth but may still scan all targets, so it is not assumed
to solve runtime.

### C. Inverted n-gram retrieval — next if exact sparse Top-K remains too slow

Build postings from informative n-grams and score only targets reached through
query postings. Control pathological postings using document-frequency
thresholds/caps and IDF weighting.

This is the first design intended to change the amount of target work
fundamentally rather than merely bounding memory.

### D. Dense ANN / FAISS — later, only if evidence justifies representation change

FAISS IVF/HNSW/PQ are non-exhaustive options for dense vectors, but adopting
them would change our currently validated sparse representation. They are not
the first optimization.

## Non-negotiable constraints

- Country remains part of retrieval partitioning.
- Source 1 is queried against Source 2/3 only.
- No external data, lookup or geocoding.
- Candidate generation is bounded.
- Candidate recall is measured before classifier work.
- Retrieval changes must be compared on fixed query/target pools.
- No full-country run before the scale ladder passes.
- Raw data and generated artifacts remain uncommitted.

## Stage checklist

### R0 — restart and freeze baseline
- [x] Create a clean optimization branch from the latest measured baseline.
- [x] Record the measured scalability failure and new architecture decision.
- [ ] Run the existing test suite unchanged.
- [ ] Record baseline candidate IDs/scores for a deterministic small fixture.

**Gate:** existing tests pass; no retrieval behavior changed.

### R1 — dependency/algorithm feasibility
- [ ] Verify candidate sparse Top-N library supports our Python version.
- [ ] Verify acceptable open-source licence.
- [ ] Build a tiny in-memory cosine Top-K experiment.
- [ ] Compare IDs and scores against brute-force exact retrieval.
- [ ] Do not edit production retrieval yet.

**Gate:** exact Top-K equivalence on deterministic fixture.

### R2 — name-only controlled benchmark
- [ ] Same 1,000 India queries and 500K targets.
- [ ] Same char 3–5 gram TF-IDF and float32.
- [ ] Same K=20.
- [ ] Measure build/vectorization time, query time, total time and peak RSS.
- [ ] Compare candidate IDs/scores with baseline.
- [ ] Measure Recall@5/10/20 on the fixed reachable ground truth.

**Gate:** no meaningful recall regression; hard K bound; materially improved
RAM and/or runtime.

### R3 — scaling ladder
- [ ] 125K targets.
- [ ] 250K targets.
- [ ] 500K targets.
- [ ] 1M only after the first three pass.
- [ ] Stop immediately if RAM approaches the machine safety budget.

**Gate:** measured scaling supports multi-million-target deployment.

### R4 — address signal
- [ ] Repeat the accepted name algorithm for address.
- [ ] Handle empty addresses explicitly.
- [ ] Benchmark independently before hybrid integration.

**Gate:** address channel passes its own RAM/runtime/recall checks.

### R5 — hybrid production retrieval
- [ ] Union exact + name + address candidates.
- [ ] Deduplicate deterministically.
- [ ] Preserve strongest channel-specific scores/features.
- [ ] Enforce final per-S1 candidate ceiling.
- [ ] Stream candidate output to partitioned Parquet.

**Gate:** robustness Recall@30 is at least consistent with the validated hybrid
baseline while meeting memory/runtime constraints.

### R6 — inverted-index experiment only if needed
- [ ] Profile n-gram document-frequency distribution.
- [ ] Define safe high-DF pruning/capping.
- [ ] Implement bounded accumulator.
- [ ] Compare recall/runtime/RAM against accepted exact sparse Top-K.

**Gate:** adopt only if it improves scalability without unacceptable recall
loss.

### R7 — downstream modeling
Only after retrieval passes:
- [ ] candidate features;
- [ ] S1-grouped validation split;
- [ ] hard negatives;
- [ ] LightGBM;
- [ ] exact macro F0.5 threshold/decision tuning;
- [ ] submission validation.

## Development rule

For every step:

1. inspect current code;
2. make one bounded change;
3. review the diff;
4. run unit tests;
5. run a small deterministic functional test;
6. run the controlled real-data benchmark;
7. inspect candidate equivalence/recall, RAM and runtime;
8. either fix the same step or advance.

Do not combine algorithm, representation, feature, and model changes in one
experiment.

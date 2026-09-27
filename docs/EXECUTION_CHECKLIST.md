# Execution Checklist and Current Status

This is the authoritative stage-gated checklist for the project.  
Rule: do **one bounded change at a time**, then review code, run unit tests, run a small real-data gate, inspect recall/runtime/RAM/output invariants, and only then advance.

## Current objective

Build a precision-oriented business entity resolution system that maps noisy Source 2 / Source 3 records to Source 1 entities and produces:

- `matching_results.tsv`
- `candidate_pairs.tsv`

Every Source 1 entity must appear exactly once in matching results. Final predicted matches must be a subset of candidate pairs.

## Hardware / operating limits

Primary machine:

- Ryzen 7 7435HS
- RTX 3050
- 16 GB DDR5
- Python 3.14.7

Working memory policy:

- Prefer CPU sparse retrieval.
- Keep normal pipeline usage comfortably below total system RAM.
- Treat ~12-13 GB total system RAM usage as a stop/review threshold.
- Write large intermediates to compressed Parquet.
- Do not keep multiple large retrieval indexes resident unless measured safe.

## Dataset audit — PASSED

### Train

| Table | Rows |
|---|---:|
| Source 1 | 2,206,821 |
| Source 2 | 5,034,616 |
| Source 3 | 5,285,603 |

Country distribution:

- Source 1: India 883,188; US 1,323,633
- Source 2: India 2,017,799; US 3,016,817
- Source 3: India 2,115,547; US 3,170,056

Missing address:

- Source 2: 168,967 (3.36%)
- Source 3: 175,916 (3.33%)

Ground truth:

- 2,206,821 Source 1 rows
- 7,638,365 positive pairs
- 123,247 singleton entities (5.58%)
- 119,157 entities with exactly 1 match
- 375,212 entities with exactly 2 matches
- 1,589,205 entities with 3+ matches

Conclusion: the task is strongly one-to-many; a best-match-only solution is invalid.

### Test

| Table | Rows |
|---|---:|
| Source 1 | 1,732,544 |
| Source 2 | 4,887,273 |
| Source 3 | 5,082,316 |

Countries:

- Source 1: France 259,452; India 809,986; US 663,106
- Source 2: France 703,378; India 2,312,565; US 1,871,330
- Source 3: France 731,615; India 2,405,000; US 1,945,701

France is open-set relative to training.

## Retrieval research — PASSED / FROZEN

### Exact baseline

Exact-rule union:

- candidate pairs: 23,324,444
- true pairs retrieved: 2,278,075
- pair recall: 29.82%

Conclusion: exact blocking alone is insufficient.

### Sparse Top-K scalability

Sparse `sparse-dot-topn` retrieval replaced brute nearest-neighbor materialization.

Measured target scaling for name retrieval:

| Targets | Time | Peak RSS |
|---:|---:|---:|
| 125K | 4.45 s | 672 MB |
| 250K | 8.53 s | 848 MB |
| 500K | 16.70 s | 1.26 GB |
| 1M | 32.98 s | 1.94 GB |

Address retrieval scaling:

| Targets | Time | Peak RSS | Recall@10 |
|---:|---:|---:|---:|
| 250K | 20.35 s | 1.40 GB | 88.56% |
| 500K | 40.13 s | 2.27 GB | 87.02% |
| 1M | 81.31 s | 3.98 GB | 86.28% |

### Hybrid benchmark

India benchmark:

- 1,000 Source 1 queries
- 500,000 targets
- 416 reachable truth pairs

Initial hybrid:

- name Top-20
- address Top-10
- final ceiling 30
- 403 / 416 truth pairs retrieved
- recall 96.875%
- peak RSS ~2.33 GB

Residual exact rules recovered 0 of the 13 misses.

Word-token complement recovered only 1 extra truth pair while materially increasing candidate volume, so it was rejected.

### Address-K sweep

| Name K | Address K | Final K | Candidates | Truth | Recall |
|---:|---:|---:|---:|---:|---:|
| 20 | 10 | 30 | 29,692 | 403/416 | 96.875% |
| 20 | 15 | 35 | 34,687 | 406/416 | 97.596% |
| 20 | 20 | 40 | 39,676 | 407/416 | 97.837% |
| 20 | 30 | 50 | 49,668 | 408/416 | 98.077% |

Decision: freeze retrieval at **name Top-20 + address Top-20 -> final Top-40**.  
Reason: K=20 is the recall/candidate-volume elbow; K=30 adds ~10K candidates per 1K queries for only one additional truth pair.

## Production retrieval — IN PROGRESS

### P0 — naive production chunking

Status: **REJECTED**

Problem: target TF-IDF was refit for every query chunk. Full runtime would be unacceptable.

### P1 — reusable sparse target index

Status: **UNIT TEST GATE PASSED**

- reusable target index introduced
- matches the validated sparse primitive on unit tests
- test suite reported 62 passed

### P2 — simultaneous name + address reusable indexes

Status: **REJECTED ON MEMORY GATE**

India target pool: 4,133,346 Source 2 + Source 3 rows.

Observed during address-index construction after name index already existed:

- Python process ~6.9 GiB
- total system memory ~12.2 / 15.4 GiB (79%)
- available memory ~3.22 GiB
- process was manually stopped before unsafe memory pressure

Conclusion: do not keep name and address target indexes resident simultaneously.

### P3 — sequential two-pass production retrieval

Status: **REAL-DATA MEMORY GATE PASSED; RUNTIME GATE FAILED — OPTIMIZATION REQUIRED**

Plan:

1. Build name Top-20 index once.
2. Stream Source 1 query chunks through it.
3. Write temporary name-candidate Parquet parts.
4. Release name index and target frame; force garbage collection.
5. Build address Top-20 index once.
6. Stream the same Source 1 chunks through it.
7. Write temporary address-candidate Parquet parts.
8. Release address index and target frame.
9. Merge corresponding name/address candidate parts.
10. Deduplicate by `(s1_id, target_id)`.
11. Apply final Top-40 bound.
12. Write final compressed candidate Parquet parts.
13. Validate:
    - no duplicate candidate pair
    - <=40 candidates per Source 1
    - expected chunk coverage
    - final files readable
    - temporary files cleaned only after successful merge

Gate after implementation:

- full unit test suite — PASSED
- small synthetic two-pass equivalence test — PASSED
- India real-data startup — PASSED
- name-index build memory observed: ~9-9.5 GB total system RAM
- after raw targets released: ~10.9-11.2 GB total system RAM
- name index build time: 92.3 s
- first three chunks: exactly 20,000 candidates per 1,000-query chunk
- query throughput observed: ~3-4 minutes per 1,000 queries
- threaded query throughput observed: ~55-60 seconds per 1,000 queries
- threaded CPU behavior: reaches ~100% during sparse multiplication, drops between chunks
- threaded RAM observed: ~11.2-11.8 GB total system RAM
- threading speedup versus prior run: roughly 3-4x
- projected India name-pass time at current 55-60 s/1K: roughly 13.5-14.7 hours
- stop threshold remains 12-13 GB total system RAM

Runtime conclusion:

- 883,188 India Source-1 rows imply ~884 chunks at chunk_size=1,000.
- At 3-4 minutes/chunk, the name pass alone projects to roughly 44-59 hours.
- Therefore P3 is memory-safe enough for the name pass but operationally too slow and must not be used for full-country generation.

Implementation notes:

- sequential name/address passes are now in production code
- only one signal target frame/index is resident at a time
- raw target frame is explicitly released immediately after each index is built
- temporary candidate parts are deleted only after their merged final part is written
- synthetic equivalence and cleanup tests were added

Code commits:

- `3af1bae` sequential two-pass production retrieval
- `f71b915` two-pass equivalence / cleanup tests
- `17982e5` release raw target frame immediately after index construction

Threading optimization result: PASSED for speedup, but full-country runtime is still too high for the available time budget.

Chunk-size sweep result: REJECTED as a meaningful optimization.

Measured on India name retrieval against 4,133,346 targets:

- 1K queries: 61.12 s total = 61.12 s / 1K
- 5K queries: 299.02 s total = 59.80 s / 1K
- 10K queries: 599.63 s total = 59.96 s / 1K
- name index build: 92.72 s

Conclusion: larger batches improve normalized throughput by only ~2%, so query batching is not the bottleneck. Do not increase production chunk size for speed.

Next controlled optimization: evaluate bounded multi-key blocking on the existing 1K / 500K recall benchmark before changing production retrieval. The goal is to reduce the target pool searched per query while measuring truth-pair block coverage first. Candidate blocking must be recall-gated before any production adoption.

Do **not** start full-country generation until the optimized runtime gate passes.

## Remaining stages

### Stage 5 — Pairwise feature engineering

Status: PENDING

Planned features, computed only for retrieved candidate pairs:

- name char/string similarities
- address char/string similarities
- token similarities
- compact-name similarity
- numeric/address-number agreement
- retrieval similarities and provenance flags
- missingness indicators
- country-consistency invariant

Gate:

- deterministic feature schema
- no leakage from ground truth
- bounded memory on a candidate sample
- null/NaN audit

### Stage 6 — Training dataset / hard negatives

Status: PENDING

Plan:

- expand ground truth to positive pairs
- join positives to retrieval candidates
- candidate-derived nonmatches become hard negatives
- split validation by Source 1 entity, never by pair

Gate:

- report candidate recall on train
- report positive/negative counts
- verify no Source 1 leakage between train/validation
- explicitly account for positives missing from retrieval

### Stage 7 — LightGBM baseline

Status: PENDING

Plan:

- train pair classifier
- no dense model unless later justified
- save model + feature list + config

Gate:

- precision / recall / F0.5 on held-out Source 1 entities
- calibration/score distribution audit
- feature importance sanity check

### Stage 8 — Retrieval improvement only if justified

Status: CONDITIONAL

Dense embeddings or other extra retrieval channels are allowed only if downstream error analysis shows candidate recall is the limiting factor and the expected gain justifies runtime/memory complexity.

### Stage 9 — Entity-level decision logic

Status: PENDING

Plan:

- allow 0 / 1 / many matches per Source 1
- sweep precision-oriented thresholds on validation
- optimize the challenge's exact macro F0.5 behavior
- treat false merges as costly

Gate:

- exact evaluation implementation
- singleton behavior verified
- no forced match per entity

### Stage 10 — Full test inference

Status: PENDING

Plan:

- preprocess test partitions
- generate candidates country by country
- compute features
- score with trained LightGBM
- apply frozen entity decision logic
- stream outputs to disk

### Stage 11 — Submission validation

Status: PENDING

Required checks:

- every test Source 1 ID exactly once in `matching_results.tsv`
- candidate rows contain only valid test IDs
- predicted matches are a subset of candidates
- no duplicate candidate IDs within an entity
- no duplicate Source 1 rows
- empty match representation correct
- required TSV columns/order correct

### Stage 12 — Cleanup / reproducibility

Status: PENDING

- freeze configs
- document commands
- remove rejected experimental paths from production entrypoints
- retain experiment notes/results
- run tests + lint
- final README methodology and reproduction steps

## Working rule from this point forward

For every change:

1. State the exact problem.
2. State the smallest proposed change.
3. Make only that change.
4. Review the diff for accidental scope expansion.
5. Run unit tests.
6. Run a bounded real-data experiment.
7. Record recall / candidate volume / runtime / RAM as applicable.
8. Mark PASS / FAIL / REVISE.
9. Update this checklist.
10. Only then move to the next stage.


### R6 — cheap blocking-key coverage diagnostic

Status: **REJECTED AS EXCLUSIVE PRODUCTION BLOCKING**

Controlled India benchmark:

- 1,000 Source-1 queries
- 500,000 targets
- 416 reachable truth pairs
- name-prefix2 coverage: 66.35%
- name-prefix3 coverage: 65.87%
- name-suffix2 coverage: 34.86%
- name-suffix3 coverage: 33.89%
- first-address-number coverage: 59.62%
- union coverage across all tested keys: 90.38%

Union target-pool sizes:

- median: 121,324.5
- p90: 131,665
- p99: 139,774
- max: 146,497
- mean: 88,224

Decision:

- reject these exact keys as an exclusive blocking layer because their 90.38% truth-pair coverage would cap recall well below the frozen 97.84% hybrid retrieval result
- the pool reduction is useful but not sufficient to justify losing ~7.5 percentage points of reachable truth-pair recall
- do not adopt this blocking scheme in production

Next controlled optimization:

- test minimum-similarity pruning on the existing validated sparse retrieval
- keep the same character TF-IDF representation and Top-K
- sweep conservative similarity thresholds and measure recall + runtime
- only adopt a threshold if recall remains effectively unchanged while runtime improves materially


### R7 — minimum-similarity pruning sweep

Status: **REJECTED AS A RUNTIME OPTIMIZATION**

Controlled India benchmark:

- 1,000 Source-1 queries
- 500,000 targets
- 416 reachable truth pairs
- frozen K values: name 20, address 20, final 40

Results:

- threshold 0.00: 39,676 hybrid candidates; 407/416 truth; 97.8365% recall; 58.16 s
- threshold 0.05: 39,676 candidates; 407/416 truth; 97.8365% recall; 60.82 s
- threshold 0.10: 39,676 candidates; 407/416 truth; 97.8365% recall; 59.14 s
- threshold 0.15: 39,664 candidates; 407/416 truth; 97.8365% recall; 59.84 s
- threshold 0.20: 39,451 candidates; 407/416 truth; 97.8365% recall; 61.59 s

Decision:

- thresholds through 0.20 preserve recall on this benchmark
- candidate reduction is negligible until 0.20
- runtime does not improve; measured times are flat/noisy and sometimes slower
- do not adopt similarity thresholding as a production speed optimization

Next controlled structural experiment:

- test source-split retrieval on the same 1K/500K benchmark
- retrieve Top-K independently from Source 2 and Source 3, then merge/cap to the same final Top-40
- measure recall and candidate volume versus the frozen combined-target baseline
- if recall is preserved, benchmark runtime/memory because each target index becomes roughly half-sized


### R8 — source-split retrieval diagnostic

Status: **REJECTED**

Controlled India benchmark:

- 1,000 Source-1 queries
- Source 2 targets: 250,000
- Source 3 targets: 250,000
- 416 reachable truth pairs

Combined-target baseline:

- 39,676 candidates
- 407/416 truth pairs
- 97.8365% recall
- 58.50 s

Source-split:

- 40,000 candidates
- 395/416 truth pairs
- 94.9519% recall
- 58.72 s total (30.82 s S2 + 27.90 s S3)

Decision:

- reject source-split retrieval
- it loses 12 reachable truth pairs (~2.88 percentage points recall)
- it provides no runtime improvement in the controlled benchmark
- keep combined-target retrieval semantics

Production implication:

The tested exact blocking, similarity pruning, larger batches, and source splitting do not solve the country-wide runtime bottleneck without unacceptable recall loss. Stop adding retrieval heuristics.

Next plan:

- freeze retrieval research at the validated 97.84% controlled recall
- proceed to feature/model development using bounded candidate samples now
- in parallel, production candidate generation remains an engineering bottleneck to revisit before full inference
- prioritize proving the classifier, thresholding, and output logic instead of spending the remaining project time exclusively on retrieval micro-optimizations


### Stage 5 — pairwise features

#### F1 deterministic feature builder

Status: **UNIT GATE PASSED**

- deterministic challenge-only pair features implemented
- feature tests passed on user environment
- no ground-truth fields enter feature construction

Next gate: bounded real-data feature audit before training.


#### F2 bounded real-data feature audit

Status: **PASSED**

India controlled gate:

- 1,000 Source-1 queries / 500,000 targets
- 39,676 candidate pairs -> 39,676 feature rows
- 19 output columns
- zero nulls in all feature/ID/label audit columns
- zero non-finite values in numeric features
- 407 positive candidate pairs / 39,269 negatives (1.026% positive)
- retrieval: 57.29 s
- feature generation: 0.447 s (~88.7k rows/s)

Decision:

- feature construction is correct and cheap relative to retrieval
- Stage 5 bounded gate passed
- freeze this initial feature set and advance to Stage 6
- next: construct labelled candidate training data with S1-grouped train/validation split and candidate hard negatives


#### Stage 6 bounded labelled training-data audit

Status: **PASSED**

India controlled gate:

- 1,000 S1 queries / 500,000 targets / 39,676 candidates
- deterministic grouped split, seed 42, requested validation fraction 20%
- zero S1 entities cross train/validation
- train: 767 entities, 30,446 pairs, 322 positives, 30,124 negatives
- validation: 233 entities, 9,230 pairs, 85 positives, 9,145 negatives
- positive candidate entities: 336
- max retrieved positives for one S1: 3
- feature/label/split construction: 0.536 s

Decision:

- Stage 6 split/label construction passes
- candidate negatives are retrieval hard negatives by construction
- keep all retrieved negatives for the first LightGBM baseline; do not downsample yet
- advance to Stage 7 baseline classifier


#### Stage 7 bounded LightGBM pair-classifier baseline

Status: **PAIR MODEL GATE PASSED; THRESHOLD NOT YET SELECTED**

India controlled validation:

- train: 30,446 candidate pairs / 322 positives
- validation: 9,230 candidate pairs / 85 positives
- early stopping best iteration: 169
- training: 0.254 s
- ROC-AUC: 0.999278
- average precision: 0.947108
- pair operating points show expected precision/recall tradeoff:
  - 0.50: P=.8875, R=.8353
  - 0.70: P=.9571, R=.7882
  - 0.80: P=.9692, R=.7412
  - 0.90: P=.9831, R=.6824
  - 0.98: P=1.0, R=.5176

Decision:

- LightGBM pair separation is strong enough to continue
- do not choose a threshold from pair-level precision/recall
- next gate must sweep thresholds using the challenge's exact per-S1 macro F0.5, including singleton entities


#### Stage 7 exact entity-level threshold sweep

Status: **METRIC GATE PASSED; THRESHOLD NEEDS ROBUSTNESS VALIDATION**

India controlled validation:

- 233 validation S1 entities
- 796 total truth pairs for those entities
- 9,230 retrieved candidate pairs
- best 0.01-grid threshold: 0.09
- best macro F0.5: 0.262520
- broad near-best region: approximately 0.09-0.23 (~0.2588-0.2625)
- 0.50 macro F0.5: 0.244611
- high pair-precision thresholds are materially worse at entity level

Interpretation:

- do not freeze 0.09 from one 233-entity validation partition
- low absolute macro score is largely constrained by candidate coverage: many validation truth pairs are outside the bounded 500k target pool and therefore impossible to predict in this controlled experiment
- next controlled gate: repeat grouped splits across fixed seeds on the same frozen candidate/features and inspect threshold stability before selecting an operating threshold


#### Stage 7 grouped-split threshold robustness

Status: **PASSED FOR OPERATING REGION; DO NOT OVERFIT PER-SPLIT OPTIMA**

Five deterministic S1-grouped splits:

- per-seed best thresholds: 0.04, 0.10, 0.09, 0.15, 0.25
- best mean threshold across seeds: 0.09
- best mean macro F0.5: 0.239285
- thresholds 0.05-0.11 form a broad near-best mean region (~0.2378-0.2393)
- high thresholds degrade consistently

Decision:

- freeze provisional operating threshold at 0.09 for the current India bounded baseline
- treat 0.09 as provisional, not globally calibrated: France is unseen in training and production candidate coverage differs from the bounded 500k experiment
- stop threshold micro-tuning
- next priority is production/end-to-end engineering and submission validation


#### Stage 8 faster name-retrieval experiment

Status: **PROMISING SPEED RESULT; RECALL COMPARISON INVALID, REVISE BENCHMARK**

India 1K queries / 500K targets:

- frozen char 3-5: 17.33s, 20,000 candidates
- word 1-2: 4.25s, 20,000 candidates
- observed speedup: 4.08x

Important benchmark flaw:

- each retriever's recall denominator was restricted to target IDs appearing in that retriever's own candidate output
- frozen truth_pool=283 while word truth_pool=258 proves the denominators differ
- therefore the reported 99.29% vs 99.22% recalls are not comparable and must not justify a production switch

Decision:

- do not replace frozen retrieval yet
- next controlled change: fix benchmark to use one common truth pool defined by the bounded target universe, then rerun


#### Stage 8 word 1-2 gram name retriever — corrected benchmark

Status: **REJECTED**

Common truth pool, India 1K queries / 500K targets:

- frozen char 3-5: 16.74s, 281/416 truth pairs, name-only recall 67.55%
- word 1-2: 4.29s, 256/416 truth pairs, name-only recall 61.54%
- speedup: 3.91x
- absolute recall loss: 6.01 percentage points

Decision:

- speed improvement is real, but recall loss is too large for a precision-heavy entity-resolution pipeline where retrieval misses are unrecoverable
- do not replace the frozen char 3-5 name retriever with word 1-2
- preserve frozen retrieval quality and move to a different production-engineering strategy


#### Stage 8 exact retrieval batch-size benchmark

Status: **REJECTED AS A RUNTIME SOLUTION**

India 10K queries / 500K targets, frozen char 3-5 Name20:

- batch 1,000: 70.08s, 142.70 queries/s
- batch 5,000: 68.39s, 146.22 queries/s, 1.025x speedup
- batch 10,000: 68.46s, 146.07 queries/s, 1.024x speedup
- all candidate outputs identical to batch-1,000 baseline

Decision:

- correctness is preserved, but ~2.4% speedup is operationally insignificant
- do not increase production batch size as the primary runtime fix
- runtime bottleneck is sparse similarity computation itself, not Python/chunk-call overhead
- next production optimization must change execution strategy while preserving the frozen candidate semantics


#### Stage 8 sparse retrieval thread scaling

Status: **PASSED — PRODUCTION THREAD COUNT = 2**

India 5K queries / 500K targets, frozen char 3-5 Top20:

- 1 thread: 20.72s, 241.27 queries/s
- 2 threads: 13.44s, 372.04 queries/s, 1.542x vs single thread
- 4 threads: 17.26s, 289.64 queries/s
- 8 threads: 27.31s, 183.10 queries/s
- 16 threads: 33.87s, 147.60 queries/s
- candidate outputs identical at every thread count

Decision:

- current n_jobs=-1 / 16-thread production setting is actively harmful on this machine
- use n_jobs=2 for production sparse retrieval
- this preserves exact frozen candidate semantics while materially improving throughput
- next gate: patch production retrieval default to 2 threads and run regression tests before any full-country execution

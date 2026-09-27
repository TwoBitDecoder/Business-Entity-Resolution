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

Status: **NEXT CHANGE**

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

- full unit test suite
- small synthetic two-pass equivalence test
- India real-data startup
- record name-index RAM
- record address-index RAM
- record first 3-5 query-chunk timings
- stop if total RAM approaches 12-13 GB

Do **not** start full-country generation until this gate passes.

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

# Production Runbook

This runbook records the validated execution order. Do not skip gates or replace the frozen retriever with rejected experimental variants.

## 1. Environment and tests

```bash
python -m pip install -r requirement.txt
python -m pip install -e .
python -m pytest
```

Raw challenge files stay under `data/train/` and `data/test/` and must not be committed.

## 2. Preprocess

Training preprocessing is available as:

```bash
python -m business_entity_resolution.preprocess
```

Expected training output: `artifacts/preprocessed/train/<source>/country=<country>/records.parquet`.

Before production inference, create the equivalent test partitions under `artifacts/preprocessed/test/` for Source 1, Source 2 and Source 3. The current preprocessing module is train-oriented, so do not assume its CLI automatically preprocesses test files.

## 3. Model artifact

The production scorer expects:

```text
artifacts/model/pair_model.joblib
artifacts/model/metadata.json
```

The model must be trained from labelled candidate features with the deterministic Source-1-grouped split. `model_artifact.train_and_save(...)` persists the classifier and metadata. The current provisional threshold is 0.09.

Do not interpret 0.09 as universally calibrated: it was selected from bounded India validation and France is open-set.

## 4. Candidate generation

Frozen semantics: char TF-IDF 3-5 grams, Name Top-20 + Address Top-20, final Top-40.

Run countries independently so progress is checkpointable:

```bash
python -m business_entity_resolution.production_candidates --root artifacts/preprocessed/test --output artifacts/candidates/test --country India
python -m business_entity_resolution.production_candidates --root artifacts/preprocessed/test --output artifacts/candidates/test --country US
python -m business_entity_resolution.production_candidates --root artifacts/preprocessed/test --output artifacts/candidates/test --country France
```

Important: candidate generation is the unresolved runtime bottleneck. The full India target benchmark projects the name pass alone at ~17.59 hours.

Temporary signal checkpoints live below `artifacts/candidates/test/_work/`. Completed signal chunks are reused after interruption and checked against the expected query chunk.

Do not use the top-level generator with destructive replacement between separate country runs. The commands above should therefore be treated cautiously until the CLI replacement behavior is revised for multi-country accumulation.

## 5. Production preflight

After all country candidate parts and the model artifact exist:

```python
from business_entity_resolution.inference import validate_production_inputs
print(validate_production_inputs())
```

This checks required partitions, model metadata, country candidate presence, and contiguous candidate-part numbering.

## 6. Checkpointed scoring

```python
from business_entity_resolution.inference import score_production_parts
print(score_production_parts())
```

Scored parts are written under `artifacts/scored/test/country=<country>/`. Existing scored parts are reused unless `replace=True`.

## 7. Assemble and validate submission

```python
from business_entity_resolution.inference import assemble_scored_submission
print(assemble_scored_submission())
```

Expected files:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
output/metadata.json
```

Assembly refuses incomplete scored checkpoint coverage. Submission validation checks:

- every test Source 1 ID appears exactly once
- only valid test target IDs are used
- candidate pairs are unique
- matched IDs contain no duplicates
- every final match is present in candidate pairs
- required column names/order are exact

## 8. Final checks

```bash
python -m pytest
python -m ruff check src tests
```

Inspect `output/metadata.json`, row counts, file sizes, and TSV headers before packaging.

## Known limitations / do not misrepresent

- Full test candidate generation has not been demonstrated within the available runtime budget.
- The provisional 0.09 threshold comes from bounded India experiments.
- France has no training-country calibration.
- The bounded smoke test proves component integration, not leaderboard performance.
- Rejected prefix/word coarse prefilters are experiment evidence, not production paths.

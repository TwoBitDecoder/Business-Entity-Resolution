# Business Entity Resolution

Offline ML pipeline for linking noisy Source 2 / Source 3 business records to deduplicated Source 1 entities. The system supports zero, one, or many matches per Source 1 entity and uses the challenge's precision-oriented macro F0.5 objective.

## Validated baseline

The current production baseline is:

- deterministic text normalization and country-partitioned Parquet
- character TF-IDF (3-5 grams), combined Source 2 + Source 3 target pool
- Name Top-20 + Address Top-20 -> deduplicated final Top-40
- engineered pairwise name/address/numeric/missingness features
- LightGBM pair classifier
- provisional decision threshold: 0.09
- checkpointed candidate generation and scoring
- strict submission validation

The controlled India 1K-query / 500K-target retrieval benchmark reached 407/416 reachable truth pairs (97.84% candidate recall). A bounded real-data end-to-end smoke test also passes.

## Important production limitation

The exact frozen retrieval semantics are quality-validated but not fast enough for a short full-scale run on the benchmark machine. On the full 4,133,346-row India target pool, the measured Name Top-20 pass processed 1,000 queries in 71.70 seconds, projecting to about 17.59 hours for India's name pass alone. Address retrieval and other countries add more work.

For that reason this repository does **not** claim that full test inference has completed. Lower-recall prefilters tested during development were rejected rather than silently promoted.

## Repository

- `src/business_entity_resolution/` — pipeline code
- `configs/` — frozen configuration
- `tests/` — unit/integration tests
- `docs/EXECUTION_CHECKLIST.md` — stage gates and experiment decisions
- `docs/PRODUCTION_RUNBOOK.md` — reproducible production procedure
- `data/` — local challenge TSVs (gitignored)
- `artifacts/` — generated Parquet/model/checkpoints (gitignored)
- `output/` — final submission files (gitignored)

No external business lookup, geocoding, or external data augmentation is used.

## Quick verification

```bash
python -m pytest
python -m business_entity_resolution.stage8_end_to_end_smoke
```

See `docs/PRODUCTION_RUNBOOK.md` before starting a long production run.

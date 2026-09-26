# Business Entity Resolution

Offline ML pipeline for linking noisy Source 2 / Source 3 business records to deduplicated Source 1 entities.

## Objective
For every Source 1 entity, predict zero, one, or many matching Source 2/3 entity IDs while prioritizing precision under macro F0.5.

## Pipeline
1. Validate and load TSV inputs.
2. Normalize names, addresses, tokens and numeric signals.
3. Generate candidates using multiple retrieval/blocking channels.
4. Build pairwise similarity and contradiction features.
5. Train a precision-oriented pair classifier.
6. Tune decisions on entity-level macro F0.5.
7. Export `candidate_pairs.tsv` and `matching_results.tsv`.

## Repository
- `src/business_entity_resolution/` — pipeline code
- `configs/` — reproducible experiment configuration
- `tests/` — unit/integration tests
- `docs/` — architecture and methodology
- `data/` — local challenge data (gitignored)
- `artifacts/` — generated models/indexes/reports (gitignored)
- `output/` — submission files (gitignored except placeholder)

No external business identity lookup or external data augmentation belongs in this project.

# Stage 4A — Preprocessing

This stage converts the raw training TSV sources into reusable country-partitioned
Parquet. It is deliberately separate from retrieval so storage correctness can
be verified before candidate-generation work begins.

Run:

```bash
python -m business_entity_resolution.preprocess
```

Outputs live under `artifacts/preprocessed/train/<source>/country=<country>/`.
The command validates input/output row counts, per-country counts, and entity-ID
uniqueness and writes `metadata.json`.

Acceptance gate: tests pass, full-data preprocessing completes without memory
pressure, metadata matches the Stage-1 audit, and elapsed time is recorded.
Do not proceed to Top-K retrieval until this gate passes.

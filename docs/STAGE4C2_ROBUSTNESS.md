# Stage 4C.2 — retrieval robustness

This checkpoint tests whether the Stage 4C.1 gain repeats outside the first
1,000 India rows.

Default experiment: two deterministic random S1 samples in India and two in
the US. Each run uses 500 queries and the already validated 500K target pool
(250K from S2 + 250K from S3). The smaller query count keeps four runs bounded.

```bash
python -m pytest
python -m business_entity_resolution.retrieval_robustness
```

Output: `artifacts/stage4c2_robustness.json`.

The report preserves target-pool coverage, name-only Recall@K, hybrid Recall@K,
candidate count, and missed reachable truth pairs. Do not increase scale until
these four runs have been inspected.

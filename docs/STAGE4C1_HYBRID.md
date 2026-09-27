# Stage 4C.1 — hybrid name + address retrieval

This experiment keeps the validated name Top-20 retriever and adds a separate
address Top-10 retrieval channel. The two candidate sets are unioned,
deduplicated, and capped at 30 candidates per S1 query.

This is deliberately an experiment, not the production retrieval architecture.
Name and address cosine scores are not calibrated against each other; the goal
is to test whether address retrieval contributes missing true positives.

Gate:

```bash
python -m pytest
python -m business_entity_resolution.hybrid_retrieval_benchmark \
  --queries 1000 --targets-per-source 250000 \
  --name-k 20 --address-k 10 --final-k 30
```

Upload `artifacts/stage4c1_hybrid.json`. Keep the method only if it materially
improves ground-truth recall over the same target pool.

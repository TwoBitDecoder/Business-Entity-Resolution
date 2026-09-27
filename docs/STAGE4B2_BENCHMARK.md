# Stage 4B.2 — controlled real-data benchmark

This checkpoint adds orchestration around the validated 4B.1 primitive, but it
still does **not** run against an entire country partition.

Default benchmark:

```bash
python -m business_entity_resolution.retrieval_benchmark_runner
```

Defaults are intentionally conservative: 1,000 S1 queries, 25,000 S2 targets,
25,000 S3 targets, Top-20. The output is
`artifacts/stage4b2_benchmark.json`.

The benchmark records candidate cardinality, the hard candidate ceiling,
runtime, throughput, and similarity distribution. This is a resource/scaling
checkpoint, not yet a ground-truth Recall@K benchmark.

Gate: pytest must pass first. Then run the default benchmark. Increase target
sample size only after inspecting runtime and memory behavior.

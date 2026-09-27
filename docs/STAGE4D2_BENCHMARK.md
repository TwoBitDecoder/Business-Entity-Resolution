# Stage 4D.2 — reusable-index runtime and memory gate

This checkpoint measures the Stage 4D.1 architecture before increasing scale.

Default workload:
- India;
- 1,000 S1 queries;
- 250K S2 + 250K S3 targets;
- 250-query chunks;
- name Top-20 + address Top-10;
- final hybrid cap 30.

The report separates target loading, name-index build, address-index build, and
query-phase time. It also records process peak RSS and verifies the per-chunk
candidate ceiling.

Run:

```bash
python -m pytest
python -m business_entity_resolution.production_benchmark
```

Upload `artifacts/stage4d2_production_benchmark.json`.

Do not increase target scale until runtime and memory are inspected.

# Stage 4D.4b — target-count scaling characterization

This is a measurement-only gate. It does **not** change retrieval, Top-K values,
vectorization, working-memory settings, or ranking.

The benchmark runs the existing Stage 4D.4a process in a **fresh Python
subprocess for each target size**. This is intentional: `ru_maxrss` is a
process-lifetime high-water mark, so reusing one process would contaminate
smaller/later measurements.

Default total target pools are:

- 125K = 62.5K from Source 2 + 62.5K from Source 3
- 250K = 125K + 125K
- 500K = 250K + 250K

Queries remain fixed at 1,000, chunk size at 250, and all retrieval parameters
remain unchanged.

Run:

```bash
python -m pytest
python -m business_entity_resolution.scaling_benchmark
```

Upload `artifacts/stage4d4b_scaling_benchmark.json`.

Gate review compares target count against index live RSS, query peak RSS,
build/query runtime, and candidate bounds. Do not scale beyond 500K until the
curve is reviewed.

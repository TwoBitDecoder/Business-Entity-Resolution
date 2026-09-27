# Stage 4D.3 — sequential-signal memory gate

Stage 4D.2 reached ~4.44 GB process max RSS with only 500K targets while both
name and address indexes were live. This checkpoint keeps retrieval semantics
unchanged but prevents the two indexes from coexisting.

Execution:
1. load the bounded target pool;
2. build/query name index;
3. delete name index and force GC;
4. build/query address index;
5. delete address index and force GC;
6. merge bounded candidate results.

The report includes **current RSS checkpoints** from Linux `/proc/self/statm`
as well as process max RSS. Max RSS is historical and cannot decrease after an
allocation, so current RSS is the important evidence that freeing the first
index actually releases live memory.

Run:

```bash
python -m pytest
python -m business_entity_resolution.sequential_retrieval_benchmark
```

Upload `artifacts/stage4d3_sequential_benchmark.json`. Keep the default 500K
target workload until this gate is reviewed.

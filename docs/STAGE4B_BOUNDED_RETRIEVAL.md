# Stage 4B.1 — bounded sparse retrieval

This checkpoint introduces only the in-memory retrieval primitive used on a
small S1 chunk inside one country partition.

It uses character 3–5 gram TF-IDF and cosine nearest-neighbour search. The
critical contract is bounded output: at most `K` candidates per non-empty
query. No full dataset runner or candidate persistence is included yet.

## Gate

Run:

```bash
python -m pytest
```

Do not run this module over the full Parquet dataset yet. After this checkpoint
passes, Stage 4B.2 will add partition/chunk orchestration with explicit resource
limits and benchmark instrumentation.

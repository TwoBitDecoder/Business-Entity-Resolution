# Stage 4B.3 — ground-truth Recall@K

This checkpoint measures whether the bounded character n-gram retriever actually
finds known positive pairs.

Important: the benchmark target pool is sampled by taking the first N records
from S2 and S3. Therefore Recall@K is computed only against ground-truth
positive IDs that are actually present in that target pool. The report also
includes target-pool truth coverage so this limitation is explicit.

## Gate

First run:

```bash
python -m pytest
```

Then run the same 500K-target scale already validated in Stage 4B.2:

```bash
python -m business_entity_resolution.retrieval_recall \
  --queries 1000 \
  --targets-per-source 250000 \
  --top-k 20
```

Upload `artifacts/stage4b3_recall.json` before increasing scale or changing
the retrieval algorithm.

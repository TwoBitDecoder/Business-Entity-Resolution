# Large-scale retrieval plan

The training audit contains 2,206,821 Source-1 rows and 7,638,365 labelled
positive pairs. Candidate retrieval is therefore benchmarked before any
pairwise classifier is trained.

## Stage 2

Run:

```bash
python -m business_entity_resolution.retrieval_benchmark
```

The command evaluates country-scoped exact normalized-name, compact-name and
exact normalized-address blocking. It reports candidate volume, positive-pair
recall and target block-size tails, and writes
`artifacts/retrieval_benchmark.json`.

Do not materialize the full candidate table until the block-size report has
been inspected. Very common names or addresses can create quadratic blocks.

## Decision gate

The next retrieval implementation should be chosen from measured results:

- If exact/compact union recall is already very high, cap pathological blocks
  and add inexpensive token/numeric rules for the residual positives.
- If recall is materially short, add sparse character n-gram retrieval in
  country partitions and benchmark Recall@K.
- Dense embeddings remain optional and should only be introduced if sparse
  retrieval leaves a meaningful recall gap.

Classifier training starts only after candidate recall and candidate volume are
both acceptable.

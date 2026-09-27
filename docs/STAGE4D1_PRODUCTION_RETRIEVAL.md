# Stage 4D.1 — reusable index + bounded query chunks

This checkpoint changes execution architecture only. Retrieval semantics remain
the validated Stage 4C design.

The experimental code refit TF-IDF and the nearest-neighbor index on every
retrieval call. Stage 4D.1 introduces a reusable `SparseTargetIndex`: build
the target vectorizer/matrix/index once for one country and one signal, then
reuse it for many S1 chunks.

It also introduces:
- bounded Parquet query iteration;
- per-chunk cardinality validation;
- hybrid union/deduplication with a hard final-K ceiling.

This checkpoint intentionally does **not** run the full multi-million-row
dataset yet. The next gate measures memory/runtime of the reusable index before
we build disk candidate output or increase scale.

Validation:

```bash
python -m pytest
```

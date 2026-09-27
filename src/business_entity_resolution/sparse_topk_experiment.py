"""R1 feasibility: exact sparse Top-K by sparse matrix multiplication.

This experiment intentionally uses only SciPy/scikit-learn dependencies already
present in the project. It is a correctness prototype, not production retrieval.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from .bounded_retrieval import RetrievalConfig


def sparse_matmul_topk(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    config: RetrievalConfig | None = None,
    text_column: str = "name_compact",
) -> pl.DataFrame:
    """Compute exact cosine Top-K from an explicitly materialized sparse product."""
    cfg = config or RetrievalConfig()
    required = {"entity_id", text_column}
    for label, frame in (("queries", queries), ("targets", targets)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    q = queries.select("entity_id", text_column).filter(pl.col(text_column) != "")
    t = targets.select("entity_id", text_column).filter(pl.col(text_column) != "")
    if q.is_empty() or t.is_empty():
        return pl.DataFrame(
            schema={"s1_id": pl.String, "target_id": pl.String,
                    "name_similarity": pl.Float32}
        )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        lowercase=False,
        dtype=np.float32,
        norm="l2",
    )
    target_matrix = vectorizer.fit_transform(t[text_column].to_list())
    query_matrix = vectorizer.transform(q[text_column].to_list())

    # L2-normalized TF-IDF means the dot product is cosine similarity.
    similarities = (query_matrix @ target_matrix.T).tocsr()
    q_ids = q["entity_id"].to_list()
    t_ids = t["entity_id"].to_list()
    k = min(cfg.top_k, t.height)
    rows: list[tuple[str, str, float]] = []

    for qi, qid in enumerate(q_ids):
        row = similarities.getrow(qi)
        if row.nnz == 0:
            continue
        scores = row.data
        cols = row.indices
        eligible = scores >= cfg.min_similarity
        scores, cols = scores[eligible], cols[eligible]
        if scores.size > k:
            chosen = np.argpartition(scores, -k)[-k:]
            scores, cols = scores[chosen], cols[chosen]
        # Deterministic order: similarity descending, then target ID ascending.
        order = sorted(range(scores.size), key=lambda i: (-float(scores[i]), t_ids[int(cols[i])]))
        for i in order:
            rows.append((qid, t_ids[int(cols[i])], float(scores[i])))

    return pl.DataFrame(
        rows, schema=["s1_id", "target_id", "name_similarity"], orient="row"
    ).with_columns(pl.col("name_similarity").cast(pl.Float32))

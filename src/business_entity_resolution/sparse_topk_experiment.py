"""R1.2: bounded exact sparse Top-K feasibility experiment.

The representation is unchanged from the validated baseline: character
3-5-gram, L2-normalized float32 TF-IDF. Only the sparse similarity operation is
changed to retain Top-N results during multiplication.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .bounded_retrieval import RetrievalConfig


def sparse_matmul_topk(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    config: RetrievalConfig | None = None,
    text_column: str = "name_compact",
) -> pl.DataFrame:
    """Compute bounded exact cosine Top-K with sparse Top-N multiplication."""
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
    target_matrix = vectorizer.fit_transform(t[text_column].to_list()).tocsr()
    query_matrix = vectorizer.transform(q[text_column].to_list()).tocsr()

    k = min(cfg.top_k, t.height)
    # L2-normalized TF-IDF dot products are cosine similarities. top_n keeps
    # the similarity output bounded instead of materializing the full product.
    similarities = sp_matmul_topn(
        query_matrix,
        target_matrix.T.tocsc(),
        top_n=k,
        threshold=cfg.min_similarity,
        sort=True,
    ).tocsr()

    q_ids = q["entity_id"].to_list()
    t_ids = t["entity_id"].to_list()
    rows: list[tuple[str, str, float]] = []
    for qi, qid in enumerate(q_ids):
        start, end = similarities.indptr[qi], similarities.indptr[qi + 1]
        cols = similarities.indices[start:end]
        scores = similarities.data[start:end]
        # Stable presentation; Top-N selection itself is performed by the library.
        order = sorted(
            range(scores.size),
            key=lambda i: (-float(scores[i]), t_ids[int(cols[i])]),
        )
        for i in order:
            rows.append((qid, t_ids[int(cols[i])], float(scores[i])))

    return pl.DataFrame(
        rows, schema=["s1_id", "target_id", "name_similarity"], orient="row"
    ).with_columns(pl.col("name_similarity").cast(pl.Float32))

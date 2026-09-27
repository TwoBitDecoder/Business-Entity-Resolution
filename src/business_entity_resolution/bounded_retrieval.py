"""Stage 4B.1: bounded sparse character n-gram retrieval.

This module deliberately operates on an explicit query chunk and returns at
most K candidates per query. It does not materialize an all-pairs matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 20
    ngram_min: int = 3
    ngram_max: int = 5
    min_similarity: float = 0.0
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("invalid n-gram range")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")


def retrieve_topk(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    config: RetrievalConfig | None = None,
    text_column: str = "name_compact",
) -> pl.DataFrame:
    """Return bounded cosine-similarity candidates for one query chunk.

    Both frames must already belong to the same country partition. Empty query
    names produce no candidates. The output cardinality is guaranteed to be
    <= non-empty-query-count * top_k.
    """
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
            schema={"s1_id": pl.String, "target_id": pl.String, "name_similarity": pl.Float32}
        )

    corpus = t[text_column].to_list()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        lowercase=False,
        dtype=np.float32,
        norm="l2",
    )
    target_matrix = vectorizer.fit_transform(corpus)
    query_matrix = vectorizer.transform(q[text_column].to_list())

    k = min(cfg.top_k, t.height)
    index = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=cfg.n_jobs,
    )
    index.fit(target_matrix)
    distances, indices = index.kneighbors(query_matrix, return_distance=True)

    q_ids = q["entity_id"].to_list()
    t_ids = t["entity_id"].to_list()
    rows: list[tuple[str, str, float]] = []
    for qi, qid in enumerate(q_ids):
        for distance, ti in zip(distances[qi], indices[qi], strict=True):
            similarity = float(1.0 - distance)
            if similarity >= cfg.min_similarity:
                rows.append((qid, t_ids[int(ti)], similarity))

    return pl.DataFrame(
        rows,
        schema=["s1_id", "target_id", "name_similarity"],
        orient="row",
    ).with_columns(pl.col("name_similarity").cast(pl.Float32))


def validate_bound(result: pl.DataFrame, query_count: int, top_k: int) -> None:
    """Fail fast if a retriever ever violates its cardinality contract."""
    if result.height > query_count * top_k:
        raise RuntimeError(
            f"candidate bound violated: {result.height} > {query_count} * {top_k}"
        )
    if result.height and result.group_by("s1_id").len()["len"].max() > top_k:
        raise RuntimeError("per-query top-k bound violated")

"""Stage 4D.1: fixed-index, chunked sparse retrieval primitives.

Unlike the experimental retriever, the target TF-IDF matrix and NN index are
built once and reused across bounded query chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .bounded_retrieval import RetrievalConfig, validate_bound


@dataclass
class SparseTargetIndex:
    """Reusable target index for one country and one text signal."""

    target_ids: list[str]
    vectorizer: TfidfVectorizer
    index: NearestNeighbors
    top_k: int
    min_similarity: float
    text_column: str

    @classmethod
    def build(
        cls,
        targets: pl.DataFrame,
        *,
        config: RetrievalConfig,
        text_column: str,
    ) -> "SparseTargetIndex":
        required = {"entity_id", text_column}
        missing = required - set(targets.columns)
        if missing:
            raise ValueError(f"targets missing columns: {sorted(missing)}")
        t = targets.select("entity_id", text_column).filter(pl.col(text_column) != "")
        if t.is_empty():
            raise ValueError(f"no non-empty targets for {text_column}")

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(config.ngram_min, config.ngram_max),
            lowercase=False,
            dtype=np.float32,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(t[text_column].to_list())
        k = min(config.top_k, t.height)
        nn = NearestNeighbors(
            n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=config.n_jobs
        )
        nn.fit(matrix)
        return cls(
            target_ids=t["entity_id"].to_list(),
            vectorizer=vectorizer,
            index=nn,
            top_k=k,
            min_similarity=config.min_similarity,
            text_column=text_column,
        )

    def query(self, queries: pl.DataFrame) -> pl.DataFrame:
        required = {"entity_id", self.text_column}
        missing = required - set(queries.columns)
        if missing:
            raise ValueError(f"queries missing columns: {sorted(missing)}")
        q = queries.select("entity_id", self.text_column).filter(
            pl.col(self.text_column) != ""
        )
        if q.is_empty():
            return pl.DataFrame(schema={
                "s1_id": pl.String, "target_id": pl.String,
                "similarity": pl.Float32,
            })

        matrix = self.vectorizer.transform(q[self.text_column].to_list())
        distances, indices = self.index.kneighbors(matrix, return_distance=True)
        rows: list[tuple[str, str, float]] = []
        for qi, qid in enumerate(q["entity_id"].to_list()):
            for distance, ti in zip(distances[qi], indices[qi], strict=True):
                similarity = float(1.0 - distance)
                if similarity >= self.min_similarity:
                    rows.append((qid, self.target_ids[int(ti)], similarity))
        result = pl.DataFrame(
            rows, schema=["s1_id", "target_id", "similarity"], orient="row"
        ).with_columns(pl.col("similarity").cast(pl.Float32))
        validate_bound(result, q.height, self.top_k)
        return result


def iter_parquet_chunks(
    path: str | Path,
    *,
    columns: tuple[str, ...],
    chunk_size: int,
) -> Iterator[pl.DataFrame]:
    """Yield bounded slices without collecting the full query partition."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    source = pl.scan_parquet(path).select(*columns)
    offset = 0
    while True:
        chunk = source.slice(offset, chunk_size).collect(engine="streaming")
        if chunk.is_empty():
            break
        yield chunk
        offset += chunk.height


def merge_hybrid(
    name: pl.DataFrame,
    address: pl.DataFrame,
    *,
    final_k: int,
) -> pl.DataFrame:
    """Union two signal results, deduplicate, and enforce final per-query K."""
    if final_k < 1:
        raise ValueError("final_k must be >= 1")
    frames = []
    if not name.is_empty():
        frames.append(name.with_columns(pl.lit("name").alias("signal")))
    if not address.is_empty():
        frames.append(address.with_columns(pl.lit("address").alias("signal")))
    if not frames:
        return pl.DataFrame(schema={
            "s1_id": pl.String, "target_id": pl.String,
            "similarity": pl.Float32, "signal": pl.String,
        })
    combined = pl.concat(frames, how="vertical")
    result = (
        combined.sort(["s1_id", "target_id", "similarity"],
                      descending=[False, False, True])
        .unique(subset=["s1_id", "target_id"], keep="first", maintain_order=True)
        .sort(["s1_id", "similarity", "target_id"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("s1_id").alias("_rank"))
        .filter(pl.col("_rank") < final_k)
        .drop("_rank")
    )
    if result.height and result.group_by("s1_id").len()["len"].max() > final_k:
        raise RuntimeError("final hybrid candidate bound violated")
    return result

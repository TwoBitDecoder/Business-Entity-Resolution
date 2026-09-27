"""Stage 4C.1: bounded hybrid name + address candidate retrieval."""

from __future__ import annotations

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk


def retrieve_hybrid_topk(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    name_k: int = 20,
    address_k: int = 10,
    final_k: int = 30,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Union bounded name/address retrieval and keep the best final_k per query."""
    name = retrieve_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=name_k, n_jobs=n_jobs),
        text_column="name_compact",
    ).rename({"name_similarity": "similarity"}).with_columns(pl.lit("name").alias("signal"))

    address = retrieve_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=address_k, n_jobs=n_jobs),
        text_column="address_norm",
    ).rename({"name_similarity": "similarity"}).with_columns(pl.lit("address").alias("signal"))

    if name.is_empty() and address.is_empty():
        return pl.DataFrame(schema={
            "s1_id": pl.String, "target_id": pl.String,
            "similarity": pl.Float32, "signal": pl.String,
        })

    # A target retrieved by both signals is retained once, using its strongest
    # score. Signal is diagnostic only at this checkpoint.
    combined = pl.concat([name, address], how="vertical")
    return (
        combined.sort(
            ["s1_id", "target_id", "similarity"],
            descending=[False, False, True],
        )
        .unique(subset=["s1_id", "target_id"], keep="first", maintain_order=True)
        .sort(["s1_id", "similarity", "target_id"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("s1_id").alias("_rank"))
        .filter(pl.col("_rank") < final_k)
        .drop("_rank")
    )

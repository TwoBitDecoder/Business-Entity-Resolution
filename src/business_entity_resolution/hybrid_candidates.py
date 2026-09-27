"""R5.1: bounded in-memory hybrid candidate union primitive.

Combines already-validated name Top-20 and address Top-10 candidate frames.
Exact-rule integration and production streaming are intentionally separate gates.
"""

from __future__ import annotations

import polars as pl


HYBRID_COLUMNS = [
    "s1_id", "target_id", "name_similarity", "address_similarity",
    "from_name", "from_address",
]


def combine_fuzzy_candidates(
    name_candidates: pl.DataFrame,
    address_candidates: pl.DataFrame,
    *,
    final_top_k: int = 30,
) -> pl.DataFrame:
    """Deduplicate name/address candidates and keep <= final_top_k per S1."""
    if final_top_k < 1:
        raise ValueError("final_top_k must be >= 1")
    required = {"s1_id", "target_id", "name_similarity"}
    for label, frame in (("name", name_candidates), ("address", address_candidates)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} candidates missing columns: {sorted(missing)}")

    name = name_candidates.select(
        "s1_id", "target_id",
        pl.col("name_similarity").cast(pl.Float32),
    ).with_columns(
        pl.col("name_similarity").alias("name_score"),
        pl.lit(0.0, dtype=pl.Float32).alias("address_score"),
        pl.lit(True).alias("from_name"),
        pl.lit(False).alias("from_address"),
    ).drop("name_similarity")

    address = address_candidates.select(
        "s1_id", "target_id",
        pl.col("name_similarity").cast(pl.Float32).alias("address_score"),
    ).with_columns(
        pl.lit(0.0, dtype=pl.Float32).alias("name_score"),
        pl.lit(False).alias("from_name"),
        pl.lit(True).alias("from_address"),
    )

    combined = pl.concat([name, address], how="diagonal").group_by(
        ["s1_id", "target_id"]
    ).agg(
        pl.col("name_score").max().alias("name_similarity"),
        pl.col("address_score").max().alias("address_similarity"),
        pl.col("from_name").any(),
        pl.col("from_address").any(),
    ).with_columns(
        pl.max_horizontal("name_similarity", "address_similarity").alias("_rank")
    ).sort(
        ["s1_id", "_rank", "name_similarity", "address_similarity", "target_id"],
        descending=[False, True, True, True, False],
    ).with_columns(
        pl.int_range(pl.len()).over("s1_id").alias("_position")
    ).filter(
        pl.col("_position") < final_top_k
    ).drop("_rank", "_position")

    return combined.select(HYBRID_COLUMNS)


def validate_hybrid_bound(result: pl.DataFrame, final_top_k: int = 30) -> None:
    if final_top_k < 1:
        raise ValueError("final_top_k must be >= 1")
    if result.is_empty():
        return
    if result.group_by("s1_id").len()["len"].max() > final_top_k:
        raise RuntimeError("hybrid per-query candidate bound violated")
    if result.select(["s1_id", "target_id"]).n_unique() != result.height:
        raise RuntimeError("hybrid candidates contain duplicate pairs")

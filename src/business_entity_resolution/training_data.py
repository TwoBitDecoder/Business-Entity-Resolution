"""Stage 6 training-data utilities.

Labels retrieved candidate pairs from challenge ground truth and creates a
deterministic Source-1-grouped split so pairs from one S1 entity can never leak
between train and validation.
"""
from __future__ import annotations

import hashlib

import polars as pl


def label_candidate_features(
    features: pl.DataFrame,
    truth_pairs: pl.DataFrame,
) -> pl.DataFrame:
    required = {"s1_id", "target_id"}
    for label, frame in (("features", features), ("truth_pairs", truth_pairs)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    truth = truth_pairs.select("s1_id", "target_id").unique().with_columns(
        pl.lit(True).alias("is_match")
    )
    return features.join(
        truth, on=["s1_id", "target_id"], how="left"
    ).with_columns(pl.col("is_match").fill_null(False))


def _validation_bucket(s1_id: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{s1_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % 10_000


def add_grouped_split(
    labelled: pl.DataFrame,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> pl.DataFrame:
    """Assign every pair for an S1 ID to exactly one deterministic split."""
    if "s1_id" not in labelled.columns:
        raise ValueError("labelled missing column: s1_id")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    cutoff = round(validation_fraction * 10_000)
    split_map = labelled.select("s1_id").unique().with_columns(
        pl.col("s1_id").map_elements(
            lambda x: "validation"
            if _validation_bucket(x, seed) < cutoff
            else "train",
            return_dtype=pl.String,
        ).alias("split")
    )
    return labelled.join(split_map, on="s1_id", how="left")


def validate_grouped_split(frame: pl.DataFrame) -> None:
    required = {"s1_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    if frame.group_by("s1_id").agg(pl.col("split").n_unique().alias("n"))["n"].max() != 1:
        raise RuntimeError("S1 entity leakage across train/validation split")
    values = set(frame["split"].unique().to_list())
    if not values <= {"train", "validation"}:
        raise RuntimeError(f"unexpected split labels: {sorted(values)}")

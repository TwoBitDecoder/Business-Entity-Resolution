"""Pairwise feature engineering for retrieved candidate records.

All features are deterministic and use only challenge-provided record fields
plus retrieval scores/provenance. Ground truth is never used here.
"""
from __future__ import annotations

import re

import polars as pl
from rapidfuzz.fuzz import ratio, token_set_ratio

_FEATURE_COLUMNS = [
    "s1_id", "target_id",
    "name_similarity", "address_similarity", "from_name", "from_address",
    "name_ratio", "name_token_set_ratio",
    "address_ratio", "address_token_set_ratio",
    "name_exact", "name_compact_exact", "address_exact",
    "name_length_ratio", "address_length_ratio",
    "address_number_jaccard", "address_number_any_overlap",
    "query_address_missing", "target_address_missing",
]

_NUM_RE = re.compile(r"\d+")


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(ratio(a, b)) / 100.0


def _token_set(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(token_set_ratio(a, b)) / 100.0


def _length_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def _number_features(a: str, b: str) -> tuple[float, bool]:
    sa, sb = set(_NUM_RE.findall(a or "")), set(_NUM_RE.findall(b or ""))
    if not sa or not sb:
        return 0.0, False
    inter = sa & sb
    return len(inter) / len(sa | sb), bool(inter)


def build_pair_features(
    candidates: pl.DataFrame,
    source1: pl.DataFrame,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Join candidate IDs to records and compute bounded pairwise features."""
    required_candidates = {
        "s1_id", "target_id", "name_similarity", "address_similarity",
        "from_name", "from_address",
    }
    missing = required_candidates - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")

    record_required = {"entity_id", "name_norm", "name_compact", "address_norm"}
    for label, frame in (("source1", source1), ("targets", targets)):
        missing = record_required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    if candidates.is_empty():
        schema = {
            "s1_id": pl.String, "target_id": pl.String,
            "name_similarity": pl.Float32, "address_similarity": pl.Float32,
            "from_name": pl.Boolean, "from_address": pl.Boolean,
        }
        return pl.DataFrame(schema=schema)

    q = source1.select(
        pl.col("entity_id").alias("s1_id"),
        pl.col("name_norm").alias("q_name"),
        pl.col("name_compact").alias("q_name_compact"),
        pl.col("address_norm").alias("q_address"),
    )
    t = targets.select(
        pl.col("entity_id").alias("target_id"),
        pl.col("name_norm").alias("t_name"),
        pl.col("name_compact").alias("t_name_compact"),
        pl.col("address_norm").alias("t_address"),
    )
    joined = candidates.join(q, on="s1_id", how="left").join(t, on="target_id", how="left")
    if joined.select(
        pl.any_horizontal(
            pl.col("q_name").is_null(),
            pl.col("t_name").is_null(),
        ).any()
    ).item():
        raise ValueError("candidate IDs missing from supplied record frames")

    rows = []
    for r in joined.iter_rows(named=True):
        qn, tn = r["q_name"] or "", r["t_name"] or ""
        qc, tc = r["q_name_compact"] or "", r["t_name_compact"] or ""
        qa, ta = r["q_address"] or "", r["t_address"] or ""
        num_jaccard, num_overlap = _number_features(qa, ta)
        rows.append({
            "s1_id": r["s1_id"],
            "target_id": r["target_id"],
            "name_similarity": r["name_similarity"],
            "address_similarity": r["address_similarity"],
            "from_name": r["from_name"],
            "from_address": r["from_address"],
            "name_ratio": _ratio(qn, tn),
            "name_token_set_ratio": _token_set(qn, tn),
            "address_ratio": _ratio(qa, ta),
            "address_token_set_ratio": _token_set(qa, ta),
            "name_exact": bool(qn and qn == tn),
            "name_compact_exact": bool(qc and qc == tc),
            "address_exact": bool(qa and qa == ta),
            "name_length_ratio": _length_ratio(qc, tc),
            "address_length_ratio": _length_ratio(qa, ta),
            "address_number_jaccard": num_jaccard,
            "address_number_any_overlap": num_overlap,
            "query_address_missing": not bool(qa),
            "target_address_missing": not bool(ta),
        })

    return pl.DataFrame(rows).select(_FEATURE_COLUMNS)

"""Stage 5 bounded real-data feature audit.

Uses the validated 1K-query / 500K-target India benchmark, generates frozen
Name20 + Address20 -> Final40 candidates, builds pair features, and records
schema, nulls, finite-value checks, label prevalence and runtime.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .features import build_pair_features
from .hybrid_candidates import combine_fuzzy_candidates
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk

NUMERIC_FEATURES = [
    "name_similarity", "address_similarity", "name_ratio",
    "name_token_set_ratio", "address_ratio", "address_token_set_ratio",
    "name_length_ratio", "address_length_ratio", "address_number_jaccard",
]


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1000,
    target_limit_per_source: int = 250000,
) -> dict:
    base = Path(root)
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(base, "source2", country), target_limit_per_source),
        _load(_partition_path(base, "source3", country), target_limit_per_source),
    ])

    print(f"{country}: retrieving bounded candidates...", flush=True)
    t0 = time.perf_counter()
    name = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=20),
        text_column="name_compact",
    )
    address = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=20),
        text_column="address_norm",
    )
    candidates = combine_fuzzy_candidates(name, address, final_top_k=40)
    retrieval_seconds = time.perf_counter() - t0

    print(f"{country}: building {candidates.height:,} pair feature rows...", flush=True)
    t0 = time.perf_counter()
    features = build_pair_features(candidates, queries, targets)
    feature_seconds = time.perf_counter() - t0

    truth = load_truth("data/train/train_ground_truth.tsv", queries["entity_id"].to_list())
    truth_pairs = truth.select("s1_id", "target_id").unique()
    labelled = features.join(
        truth_pairs.with_columns(pl.lit(True).alias("is_match")),
        on=["s1_id", "target_id"], how="left",
    ).with_columns(pl.col("is_match").fill_null(False))

    null_counts = {c: labelled[c].null_count() for c in labelled.columns}
    nonfinite_counts = {}
    for c in NUMERIC_FEATURES:
        values = labelled[c].to_list()
        nonfinite_counts[c] = sum(
            1 for v in values if v is not None and not math.isfinite(float(v))
        )

    positives = int(labelled["is_match"].sum())
    negatives = labelled.height - positives
    result = {
        "purpose": "Stage 5 bounded real-data feature audit",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "candidate_rows": candidates.height,
        "feature_rows": features.height,
        "feature_columns": features.columns,
        "feature_column_count": len(features.columns),
        "null_counts": null_counts,
        "nonfinite_counts": nonfinite_counts,
        "positive_candidate_pairs": positives,
        "negative_candidate_pairs": negatives,
        "positive_rate": positives / labelled.height if labelled.height else 0.0,
        "retrieval_seconds": retrieval_seconds,
        "feature_seconds": feature_seconds,
        "rows_per_feature_second": (
            features.height / feature_seconds if feature_seconds else None
        ),
    }
    out = Path("artifacts") / "stage5_feature_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()

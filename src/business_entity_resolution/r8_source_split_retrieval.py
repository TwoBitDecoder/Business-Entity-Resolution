"""R8: source-split retrieval diagnostic.

Tests whether querying Source 2 and Source 3 independently can preserve the
frozen hybrid recall while shrinking each target index roughly in half.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .r5_hybrid_benchmark import _load, _pair_recall
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk


def _retrieve_one_source(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    name_k: int = 20,
    address_k: int = 20,
) -> tuple[pl.DataFrame, float]:
    t0 = time.perf_counter()
    name = sparse_matmul_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=name_k),
        text_column="name_compact",
    )
    address = sparse_matmul_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=address_k),
        text_column="address_norm",
    )
    hybrid = combine_fuzzy_candidates(name, address, final_top_k=name_k + address_k)
    return hybrid, time.perf_counter() - t0


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1000,
    target_limit_per_source: int = 250000,
) -> dict:
    base = Path(root)
    queries = _load(_partition_path(base, "source1", country), query_count)
    s2 = _load(_partition_path(base, "source2", country), target_limit_per_source)
    s3 = _load(_partition_path(base, "source3", country), target_limit_per_source)
    combined_targets = pl.concat([s2, s3], how="vertical")

    truth = load_truth("data/train/train_ground_truth.tsv", queries["entity_id"].to_list())
    truth = truth.join(
        combined_targets.select(pl.col("entity_id").alias("target_id")).unique(),
        on="target_id",
        how="inner",
    )

    # Frozen combined-target baseline.
    t0 = time.perf_counter()
    base_name = sparse_matmul_topk(
        queries,
        combined_targets,
        config=RetrievalConfig(top_k=20),
        text_column="name_compact",
    )
    base_address = sparse_matmul_topk(
        queries,
        combined_targets,
        config=RetrievalConfig(top_k=20),
        text_column="address_norm",
    )
    baseline = combine_fuzzy_candidates(base_name, base_address, final_top_k=40)
    baseline_seconds = time.perf_counter() - t0
    baseline_recall = _pair_recall(baseline, truth)

    s2_hybrid, s2_seconds = _retrieve_one_source(queries, s2)
    s3_hybrid, s3_seconds = _retrieve_one_source(queries, s3)

    # Source-local TF-IDF scores are not guaranteed globally calibrated, so
    # this is intentionally a measured diagnostic rather than an assumption.
    split_name = pl.concat(
        [
            s2_hybrid.select("s1_id", "target_id", pl.col("name_similarity")),
            s3_hybrid.select("s1_id", "target_id", pl.col("name_similarity")),
        ],
        how="vertical",
    )
    split_address = pl.concat(
        [
            s2_hybrid.select(
                "s1_id",
                "target_id",
                pl.col("address_similarity").alias("name_similarity"),
            ),
            s3_hybrid.select(
                "s1_id",
                "target_id",
                pl.col("address_similarity").alias("name_similarity"),
            ),
        ],
        how="vertical",
    )
    split = combine_fuzzy_candidates(split_name, split_address, final_top_k=40)
    split_recall = _pair_recall(split, truth)

    return {
        "purpose": "R8 source-split retrieval diagnostic",
        "country": country,
        "query_rows": queries.height,
        "targets_source2": s2.height,
        "targets_source3": s3.height,
        "reachable_truth_pairs": truth.height,
        "baseline": {
            "candidate_rows": baseline.height,
            "truth_pairs_retrieved": round((baseline_recall or 0) * truth.height),
            "pair_recall": baseline_recall,
            "retrieval_seconds": baseline_seconds,
        },
        "source_split": {
            "candidate_rows": split.height,
            "truth_pairs_retrieved": round((split_recall or 0) * truth.height),
            "pair_recall": split_recall,
            "source2_seconds": s2_seconds,
            "source3_seconds": s3_seconds,
            "retrieval_seconds": s2_seconds + s3_seconds,
        },
    }


def main() -> None:
    result = run()
    out = Path("artifacts") / "r8_source_split_retrieval.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

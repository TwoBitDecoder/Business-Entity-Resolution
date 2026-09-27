"""R5.3: diagnose truth pairs missed by the validated fuzzy hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk


def exact_pair_flags(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Return exact normalized-field flags without many-to-many exact joins."""
    q = queries.select(
        pl.col("entity_id").alias("s1_id"), "name_norm", "name_compact", "address_norm"
    )
    t = targets.select(
        pl.col("entity_id").alias("target_id"),
        pl.col("name_norm").alias("target_name_norm"),
        pl.col("name_compact").alias("target_name_compact"),
        pl.col("address_norm").alias("target_address_norm"),
    )
    return q.join(t, how="cross").with_columns(
        ((pl.col("name_norm") != "") & (pl.col("name_norm") == pl.col("target_name_norm")))
        .alias("exact_name"),
        ((pl.col("name_compact") != "") & (pl.col("name_compact") == pl.col("target_name_compact")))
        .alias("exact_compact_name"),
        ((pl.col("address_norm") != "") & (pl.col("address_norm") == pl.col("target_address_norm")))
        .alias("exact_address"),
    )


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_norm", "name_compact", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def run_residual_analysis(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *, country: str = "India", query_count: int = 1_000,
    target_limit_per_source: int = 250_000,
    name_top_k: int = 20, address_top_k: int = 10, final_top_k: int = 30,
) -> dict:
    base = Path(root)
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(base, "source2", country), target_limit_per_source),
        _load(_partition_path(base, "source3", country), target_limit_per_source),
    ], how="vertical")

    name = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=name_top_k),
        text_column="name_compact",
    )
    address = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=address_top_k),
        text_column="address_norm",
    )
    hybrid = combine_fuzzy_candidates(name, address, final_top_k=final_top_k)

    truth_all = load_truth(ground_truth_path, queries["entity_id"].to_list())
    target_ids = targets.select(pl.col("entity_id").alias("target_id")).unique()
    truth_pool = truth_all.join(target_ids, on="target_id", how="inner")
    retrieved = hybrid.select("s1_id", "target_id").unique()
    misses = truth_pool.join(retrieved, on=["s1_id", "target_id"], how="anti")

    # Only join the small set of known truth misses back to source fields. This
    # avoids recreating the unsafe many-to-many exact-blocking cross product.
    q_fields = queries.select(
        pl.col("entity_id").alias("s1_id"), "name_norm", "name_compact", "address_norm"
    )
    t_fields = targets.select(
        pl.col("entity_id").alias("target_id"),
        pl.col("name_norm").alias("target_name_norm"),
        pl.col("name_compact").alias("target_name_compact"),
        pl.col("address_norm").alias("target_address_norm"),
    )
    detail = misses.join(q_fields, on="s1_id").join(t_fields, on="target_id").with_columns(
        ((pl.col("name_norm") != "") & (pl.col("name_norm") == pl.col("target_name_norm")))
        .alias("exact_name"),
        ((pl.col("name_compact") != "") & (pl.col("name_compact") == pl.col("target_name_compact")))
        .alias("exact_compact_name"),
        ((pl.col("address_norm") != "") & (pl.col("address_norm") == pl.col("target_address_norm")))
        .alias("exact_address"),
        (pl.col("address_norm") == "").alias("query_address_empty"),
        (pl.col("target_address_norm") == "").alias("target_address_empty"),
    ).with_columns(
        pl.any_horizontal("exact_name", "exact_compact_name", "exact_address")
        .alias("recoverable_by_exact")
    )

    def count(expr: pl.Expr) -> int:
        return detail.filter(expr).height

    return {
        "purpose": "R5.3 fuzzy-hybrid residual truth analysis",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "truth_pairs_in_target_pool": truth_pool.height,
        "hybrid_truth_pairs_retrieved": truth_pool.height - detail.height,
        "missed_truth_pairs": detail.height,
        "missed_truth_entities": detail["s1_id"].n_unique() if detail.height else 0,
        "miss_characteristics": {
            "exact_name": count(pl.col("exact_name")),
            "exact_compact_name": count(pl.col("exact_compact_name")),
            "exact_address": count(pl.col("exact_address")),
            "recoverable_by_any_exact_rule": count(pl.col("recoverable_by_exact")),
            "query_address_empty": count(pl.col("query_address_empty")),
            "target_address_empty": count(pl.col("target_address_empty")),
        },
        "residual_pairs": detail.select(
            "s1_id", "target_id", "exact_name", "exact_compact_name", "exact_address",
            "recoverable_by_exact", "query_address_empty", "target_address_empty",
        ).to_dicts(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    a = p.parse_args()
    result = run_residual_analysis(
        country=a.country, query_count=a.queries,
        target_limit_per_source=a.targets_per_source,
    )
    out = Path("artifacts/retrieval_v2_r5_residual_analysis.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

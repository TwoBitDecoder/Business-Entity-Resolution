"""R6: blocking-key coverage diagnostic before production adoption.

Measures how often reachable truth pairs share cheap exact blocking keys and
estimates per-query target-pool sizes on the same controlled 1K/500K India
benchmark. This does not change production retrieval.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import polars as pl

from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth


KEYS = (
    "name_prefix2",
    "name_prefix3",
    "name_suffix2",
    "name_suffix3",
    "address_first_number",
)


def _with_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("name_compact").str.slice(0, 2).alias("name_prefix2"),
        pl.col("name_compact").str.slice(0, 3).alias("name_prefix3"),
        pl.col("name_compact").str.slice(-2, 2).alias("name_suffix2"),
        pl.col("name_compact").str.slice(-3, 3).alias("name_suffix3"),
        pl.col("address_numbers").str.split("|").list.first().fill_null("").alias("address_first_number"),
    )


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_numbers")
        .head(limit)
        .collect(engine="streaming")
    )


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1000,
    target_limit_per_source: int = 250000,
) -> dict:
    root = Path(root)
    q = _with_keys(_load(_partition_path(root, "source1", country), query_count))
    targets = _with_keys(
        pl.concat(
            [
                _load(_partition_path(root, "source2", country), target_limit_per_source),
                _load(_partition_path(root, "source3", country), target_limit_per_source),
            ],
            how="vertical",
        )
    )

    truth = load_truth("data/train/train_ground_truth.tsv", q["entity_id"].to_list())
    truth = truth.join(
        targets.select(pl.col("entity_id").alias("target_id")).unique(),
        on="target_id",
        how="inner",
    )

    qk = q.select(
        pl.col("entity_id").alias("s1_id"),
        *KEYS,
    )
    tk = targets.select(
        pl.col("entity_id").alias("target_id"),
        *KEYS,
    )
    paired = truth.join(qk, on="s1_id", how="left", suffix="_q").join(
        tk, on="target_id", how="left", suffix="_t"
    )

    coverage = {}
    for key in KEYS:
        qcol = pl.col(key)
        tcol = pl.col(f"{key}_t")
        shared = paired.filter((qcol != "") & (qcol == tcol)).height
        coverage[key] = {
            "truth_pairs_shared": shared,
            "truth_pair_coverage": shared / paired.height if paired.height else None,
        }

    any_expr = None
    for key in KEYS:
        expr = (pl.col(key) != "") & (pl.col(key) == pl.col(f"{key}_t"))
        any_expr = expr if any_expr is None else (any_expr | expr)
    any_shared = paired.filter(any_expr).height if any_expr is not None else 0

    # Estimate bounded target pools from the union of exact-key buckets.
    indexes = {key: defaultdict(set) for key in KEYS}
    target_ids = targets["entity_id"].to_list()
    for key in KEYS:
        vals = targets[key].to_list()
        idx = indexes[key]
        for tid, value in zip(target_ids, vals, strict=True):
            if value:
                idx[value].add(tid)

    pool_sizes = []
    for row in q.select(KEYS).iter_rows(named=True):
        pool = set()
        for key in KEYS:
            value = row[key]
            if value:
                pool.update(indexes[key].get(value, ()))
        pool_sizes.append(len(pool))

    s = pl.Series("pool_size", pool_sizes)
    return {
        "purpose": "R6 cheap blocking-key coverage diagnostic",
        "country": country,
        "query_rows": q.height,
        "target_rows": targets.height,
        "reachable_truth_pairs": paired.height,
        "key_coverage": coverage,
        "union": {
            "truth_pairs_shared": any_shared,
            "truth_pair_coverage": any_shared / paired.height if paired.height else None,
        },
        "target_pool_size": {
            "min": int(s.min()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "p90": float(s.quantile(0.90, interpolation="nearest")) if len(s) else None,
            "p99": float(s.quantile(0.99, interpolation="nearest")) if len(s) else None,
            "max": int(s.max()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
        },
    }


def main() -> None:
    result = run()
    out = Path("artifacts") / "r6_blocking_coverage.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

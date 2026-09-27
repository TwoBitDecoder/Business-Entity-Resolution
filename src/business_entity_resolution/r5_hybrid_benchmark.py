"""R5.2: controlled real-data benchmark for fuzzy hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates, validate_hybrid_bound
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def _pair_recall(candidates: pl.DataFrame, truth: pl.DataFrame) -> float | None:
    if truth.is_empty():
        return None
    hits = (
        candidates.select("s1_id", "target_id").unique()
        .join(truth, on=["s1_id", "target_id"], how="inner")
        .height
    )
    return hits / truth.height


def run_hybrid_benchmark(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *, country: str = "India", query_count: int = 1_000,
    target_limit_per_source: int = 250_000,
    name_top_k: int = 20, address_top_k: int = 10, final_top_k: int = 30,
) -> dict:
    base = Path(root)
    started = time.perf_counter()
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(base, "source2", country), target_limit_per_source),
        _load(_partition_path(base, "source3", country), target_limit_per_source),
    ], how="vertical")
    load_s = time.perf_counter() - started

    t = time.perf_counter()
    name = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=name_top_k),
        text_column="name_compact",
    )
    name_s = time.perf_counter() - t

    t = time.perf_counter()
    address = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=address_top_k),
        text_column="address_norm",
    )
    address_s = time.perf_counter() - t

    t = time.perf_counter()
    hybrid = combine_fuzzy_candidates(
        name, address, final_top_k=final_top_k
    )
    combine_s = time.perf_counter() - t
    validate_hybrid_bound(hybrid, final_top_k)

    query_ids = queries["entity_id"].to_list()
    truth_all = load_truth(ground_truth_path, query_ids)
    target_ids = targets.select(pl.col("entity_id").alias("target_id")).unique()
    truth_pool = truth_all.join(target_ids, on="target_id", how="inner")

    max_per_query = (
        int(hybrid.group_by("s1_id").len()["len"].max()) if hybrid.height else 0
    )
    return {
        "purpose": "R5.2 fuzzy hybrid real-data benchmark",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "settings": {
            "name_top_k": name_top_k,
            "address_top_k": address_top_k,
            "final_top_k": final_top_k,
        },
        "candidate_rows": {
            "name": name.height,
            "address": address.height,
            "hybrid": hybrid.height,
            "max_hybrid_per_query": max_per_query,
        },
        "truth_pairs_for_queries": truth_all.height,
        "truth_pairs_in_target_pool": truth_pool.height,
        "pair_recall": {
            "name": _pair_recall(name, truth_pool),
            "address": _pair_recall(address, truth_pool),
            "hybrid": _pair_recall(hybrid, truth_pool),
        },
        "timing_seconds": {
            "load": load_s,
            "name_retrieval": name_s,
            "address_retrieval": address_s,
            "combine": combine_s,
            "total": time.perf_counter() - started,
        },
        "process_max_rss_mb": _max_rss_mb(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    p.add_argument("--name-top-k", type=int, default=20)
    p.add_argument("--address-top-k", type=int, default=10)
    p.add_argument("--final-top-k", type=int, default=30)
    a = p.parse_args()
    result = run_hybrid_benchmark(
        country=a.country, query_count=a.queries,
        target_limit_per_source=a.targets_per_source,
        name_top_k=a.name_top_k, address_top_k=a.address_top_k,
        final_top_k=a.final_top_k,
    )
    out = Path("artifacts/retrieval_v2_r5_hybrid_benchmark.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

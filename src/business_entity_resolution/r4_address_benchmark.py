"""R4.2: controlled real-data address-only Sparse Top-N benchmark."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig, validate_bound
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth, recall_at_k
from .sparse_topk_experiment import sparse_matmul_topk


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def run_address_benchmark(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *, country: str = "India", query_count: int = 1_000,
    target_limit_per_source: int = 62_500, top_k: int = 10,
) -> dict:
    base = Path(root)
    started = time.perf_counter()
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(base, "source2", country), target_limit_per_source),
        _load(_partition_path(base, "source3", country), target_limit_per_source),
    ], how="vertical")
    load_s = time.perf_counter() - started

    nonempty_queries = queries.filter(pl.col("address_norm") != "").height
    nonempty_targets = targets.filter(pl.col("address_norm") != "").height
    cfg = RetrievalConfig(top_k=top_k)
    t = time.perf_counter()
    candidates = sparse_matmul_topk(
        queries, targets, config=cfg, text_column="address_norm"
    )
    retrieval_s = time.perf_counter() - t
    validate_bound(candidates, nonempty_queries, top_k)

    query_ids = queries["entity_id"].to_list()
    truth_all = load_truth(ground_truth_path, query_ids)
    target_ids = targets.select(pl.col("entity_id").alias("target_id")).unique()
    truth_pool = truth_all.join(target_ids, on="target_id", how="inner")
    metrics = recall_at_k(
        candidates, truth_pool, ks=tuple(k for k in (5, 10) if k <= top_k)
    )
    max_per_query = (
        int(candidates.group_by("s1_id").len()["len"].max())
        if candidates.height else 0
    )
    return {
        "purpose": "R4.2 address-only Sparse Top-N real-data benchmark",
        "country": country,
        "query_rows": queries.height,
        "nonempty_query_addresses": nonempty_queries,
        "target_rows": targets.height,
        "nonempty_target_addresses": nonempty_targets,
        "top_k": top_k,
        "candidate_rows": candidates.height,
        "max_candidates_per_query": max_per_query,
        "truth_pairs_for_queries": truth_all.height,
        "truth_pairs_in_target_pool": truth_pool.height,
        **metrics,
        "timing_seconds": {
            "load": load_s,
            "retrieval": retrieval_s,
            "total": time.perf_counter() - started,
        },
        "process_max_rss_mb": _max_rss_mb(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=62500)
    p.add_argument("--top-k", type=int, default=10)
    a = p.parse_args()
    result = run_address_benchmark(
        country=a.country, query_count=a.queries,
        target_limit_per_source=a.targets_per_source, top_k=a.top_k,
    )
    out = Path("artifacts/retrieval_v2_r4_address_benchmark.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

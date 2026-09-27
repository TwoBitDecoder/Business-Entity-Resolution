"""Stage 4D.2: bounded runtime and peak-RSS benchmark for reusable indexes."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .production_retrieval import SparseTargetIndex, merge_hybrid
from .retrieval_benchmark_runner import _partition_path


def _rss_mb() -> float:
    # Linux ru_maxrss is KiB; this benchmark targets the project's Linux/WSL workflow.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def run_production_benchmark(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1_000,
    target_limit_per_source: int = 250_000,
    chunk_size: int = 250,
    name_k: int = 20,
    address_k: int = 10,
    final_k: int = 30,
    n_jobs: int = -1,
) -> dict:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    root = Path(root)
    started = time.perf_counter()
    rss_start = _rss_mb()
    queries = _load(_partition_path(root, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(root, "source2", country), target_limit_per_source),
        _load(_partition_path(root, "source3", country), target_limit_per_source),
    ], how="vertical")
    loaded_s = time.perf_counter() - started
    rss_loaded = _rss_mb()

    t = time.perf_counter()
    name_index = SparseTargetIndex.build(
        targets.select("entity_id", "name_compact"),
        config=RetrievalConfig(top_k=name_k, n_jobs=n_jobs),
        text_column="name_compact",
    )
    name_build_s = time.perf_counter() - t
    rss_name = _rss_mb()

    t = time.perf_counter()
    address_index = SparseTargetIndex.build(
        targets.select("entity_id", "address_norm"),
        config=RetrievalConfig(top_k=address_k, n_jobs=n_jobs),
        text_column="address_norm",
    )
    address_build_s = time.perf_counter() - t
    rss_address = _rss_mb()

    candidate_rows = 0
    max_chunk_candidates = 0
    query_s = 0.0
    chunks = 0
    for offset in range(0, queries.height, chunk_size):
        chunk = queries.slice(offset, chunk_size)
        t = time.perf_counter()
        hybrid = merge_hybrid(
            name_index.query(chunk.select("entity_id", "name_compact")),
            address_index.query(chunk.select("entity_id", "address_norm")),
            final_k=final_k,
        )
        query_s += time.perf_counter() - t
        candidate_rows += hybrid.height
        max_chunk_candidates = max(max_chunk_candidates, hybrid.height)
        chunks += 1

    total_s = time.perf_counter() - started
    peak_rss = _rss_mb()
    return {
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "chunk_size": chunk_size,
        "chunks": chunks,
        "name_k": name_k,
        "address_k": address_k,
        "final_k": final_k,
        "candidate_rows": candidate_rows,
        "max_chunk_candidates": max_chunk_candidates,
        "max_allowed_chunk_candidates": chunk_size * final_k,
        "timing_seconds": {
            "load": loaded_s,
            "name_index_build": name_build_s,
            "address_index_build": address_build_s,
            "query_all_chunks": query_s,
            "total": total_s,
        },
        "queries_per_second_query_phase": queries.height / query_s if query_s else None,
        "rss_mb": {
            "start": rss_start,
            "after_load": rss_loaded,
            "after_name_index": rss_name,
            "after_address_index": rss_address,
            "peak": peak_rss,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    result = run_production_benchmark(
        country=a.country, query_count=a.queries,
        target_limit_per_source=a.targets_per_source,
        chunk_size=a.chunk_size, n_jobs=a.n_jobs,
    )
    out = Path("artifacts/stage4d2_production_benchmark.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

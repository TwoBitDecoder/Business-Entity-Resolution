"""Stage 4D.4a: diagnose which sequential-retrieval phase sets peak RSS."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .production_retrieval import SparseTargetIndex, merge_hybrid
from .retrieval_benchmark_runner import _partition_path


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _current_rss_mb() -> float | None:
    # Linux exposes current resident pages without another dependency.
    try:
        fields = Path("/proc/self/statm").read_text().split()
        pages = int(fields[1])
        return pages * resource.getpagesize() / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def _retrieve_signal(
    index: SparseTargetIndex,
    queries: pl.DataFrame,
    *,
    text_column: str,
    chunk_size: int,
) -> tuple[pl.DataFrame, float]:
    pieces: list[pl.DataFrame] = []
    started = time.perf_counter()
    for offset in range(0, queries.height, chunk_size):
        pieces.append(index.query(
            queries.slice(offset, chunk_size).select("entity_id", text_column)
        ))
    elapsed = time.perf_counter() - started
    if not pieces:
        return pl.DataFrame(schema={
            "s1_id": pl.String, "target_id": pl.String, "similarity": pl.Float32,
        }), elapsed
    return pl.concat(pieces, how="vertical"), elapsed


def run_sequential_benchmark(
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
    queries = _load(_partition_path(root, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(root, "source2", country), target_limit_per_source),
        _load(_partition_path(root, "source3", country), target_limit_per_source),
    ], how="vertical")
    load_s = time.perf_counter() - started
    rss_after_load = _current_rss_mb()
    max_after_load = _max_rss_mb()

    t = time.perf_counter()
    name_index = SparseTargetIndex.build(
        targets.select("entity_id", "name_compact"),
        config=RetrievalConfig(top_k=name_k, n_jobs=n_jobs),
        text_column="name_compact",
    )
    name_build_s = time.perf_counter() - t
    rss_name_index = _current_rss_mb()
    max_after_name_index = _max_rss_mb()

    name_candidates, name_query_s = _retrieve_signal(
        name_index, queries, text_column="name_compact", chunk_size=chunk_size
    )
    rss_after_name_query = _current_rss_mb()
    max_after_name_query = _max_rss_mb()

    del name_index
    gc.collect()
    rss_after_name_free = _current_rss_mb()
    max_after_name_free = _max_rss_mb()

    t = time.perf_counter()
    address_index = SparseTargetIndex.build(
        targets.select("entity_id", "address_norm"),
        config=RetrievalConfig(top_k=address_k, n_jobs=n_jobs),
        text_column="address_norm",
    )
    address_build_s = time.perf_counter() - t
    rss_address_index = _current_rss_mb()
    max_after_address_index = _max_rss_mb()

    address_candidates, address_query_s = _retrieve_signal(
        address_index, queries, text_column="address_norm", chunk_size=chunk_size
    )
    rss_after_address_query = _current_rss_mb()
    max_after_address_query = _max_rss_mb()

    del address_index
    gc.collect()
    rss_after_address_free = _current_rss_mb()
    max_after_address_free = _max_rss_mb()

    t = time.perf_counter()
    hybrid = merge_hybrid(name_candidates, address_candidates, final_k=final_k)
    merge_s = time.perf_counter() - t
    max_per_query = (
        int(hybrid.group_by("s1_id").len()["len"].max()) if hybrid.height else 0
    )
    if max_per_query > final_k:
        raise RuntimeError("final candidate bound violated")

    total_s = time.perf_counter() - started
    return {
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "chunk_size": chunk_size,
        "name_k": name_k,
        "address_k": address_k,
        "final_k": final_k,
        "name_candidate_rows": name_candidates.height,
        "address_candidate_rows": address_candidates.height,
        "candidate_rows": hybrid.height,
        "max_candidates_per_query": max_per_query,
        "timing_seconds": {
            "load": load_s,
            "name_index_build": name_build_s,
            "name_query": name_query_s,
            "address_index_build": address_build_s,
            "address_query": address_query_s,
            "merge": merge_s,
            "total": total_s,
        },
        "current_rss_mb": {
            "after_load": rss_after_load,
            "name_index": rss_name_index,
            "after_name_query": rss_after_name_query,
            "after_name_free": rss_after_name_free,
            "address_index": rss_address_index,
            "after_address_query": rss_after_address_query,
            "after_address_free": rss_after_address_free,
        },
        "max_rss_checkpoints_mb": {
            "after_load": max_after_load,
            "after_name_index": max_after_name_index,
            "after_name_query": max_after_name_query,
            "after_name_free": max_after_name_free,
            "after_address_index": max_after_address_index,
            "after_address_query": max_after_address_query,
            "after_address_free": max_after_address_free,
        },
        "process_max_rss_mb": _max_rss_mb(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    result = run_sequential_benchmark(
        country=a.country, query_count=a.queries,
        target_limit_per_source=a.targets_per_source,
        chunk_size=a.chunk_size, n_jobs=a.n_jobs,
    )
    out = Path("artifacts/stage4d4_phase_memory_diagnostic.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

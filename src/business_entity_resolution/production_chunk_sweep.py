"""Benchmark reusable production retrieval at larger query batch sizes.

Builds the India (or selected country) name index once, then measures query
throughput for several chunk sizes without changing retrieval semantics.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .production_candidates import NAME_K, _load_signal_targets
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    chunk_sizes: tuple[int, ...] = (1000, 5000, 10000),
) -> dict:
    base = Path(root)
    qpath = _partition_path(base, "source1", country)
    qrows = pl.scan_parquet(qpath).select(pl.len()).collect(engine="streaming").item()

    print(f"{country}: loading name targets...", flush=True)
    targets = _load_signal_targets(base, country, "name_compact")
    print(f"{country}: loaded {targets.height:,} targets; building name index...", flush=True)

    started = time.perf_counter()
    index = SparseTopKIndex(
        targets,
        text_column="name_compact",
        config=RetrievalConfig(top_k=NAME_K),
    )
    index_seconds = time.perf_counter() - started
    del targets
    gc.collect()
    print(f"{country}: name index ready in {index_seconds:.1f}s", flush=True)

    results = []
    for size in chunk_sizes:
        if size < 1:
            raise ValueError("chunk sizes must be >= 1")
        q = (
            pl.scan_parquet(qpath)
            .select("entity_id", "name_compact")
            .head(min(size, qrows))
            .collect(engine="streaming")
        )
        t0 = time.perf_counter()
        candidates = index.query(q)
        elapsed = time.perf_counter() - t0
        per_1000 = elapsed / q.height * 1000 if q.height else None
        row = {
            "chunk_size": q.height,
            "candidate_rows": candidates.height,
            "elapsed_seconds": elapsed,
            "seconds_per_1000_queries": per_1000,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    return {
        "purpose": "production name-query chunk-size throughput sweep",
        "country": country,
        "target_rows": len(index.target_ids),
        "name_top_k": NAME_K,
        "index_seconds": index_seconds,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    args = parser.parse_args()

    result = run(country=args.country, chunk_sizes=tuple(args.chunk_sizes))
    out = Path("artifacts") / "production_chunk_sweep.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

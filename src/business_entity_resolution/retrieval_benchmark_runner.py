"""Stage 4B.2: controlled real-data benchmark for bounded retrieval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk, validate_bound


def _partition_path(root: Path, source: str, country: str) -> Path:
    path = root / source / f"country={country}" / "records.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_head(path: Path, limit: int) -> pl.DataFrame:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact")
        .head(limit)
        .collect(engine="streaming")
    )


def run_benchmark(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_limit: int = 1_000,
    target_limit_per_source: int = 25_000,
    top_k: int = 20,
    n_jobs: int = -1,
) -> dict:
    """Benchmark one bounded query chunk against bounded S2+S3 target samples."""
    base = Path(root)
    queries = load_head(_partition_path(base, "source1", country), query_limit)
    s2 = load_head(_partition_path(base, "source2", country), target_limit_per_source)
    s3 = load_head(_partition_path(base, "source3", country), target_limit_per_source)
    targets = pl.concat([s2, s3], how="vertical")

    cfg = RetrievalConfig(top_k=top_k, n_jobs=n_jobs)
    started = time.perf_counter()
    candidates = retrieve_topk(queries, targets, config=cfg)
    elapsed = time.perf_counter() - started
    nonempty_queries = queries.filter(pl.col("name_compact") != "").height
    validate_bound(candidates, nonempty_queries, top_k)

    sims = candidates["name_similarity"]
    stats = {
        "country": country,
        "query_rows": queries.height,
        "nonempty_queries": nonempty_queries,
        "target_rows": targets.height,
        "top_k": top_k,
        "candidate_rows": candidates.height,
        "max_allowed_candidates": nonempty_queries * top_k,
        "elapsed_seconds": round(elapsed, 4),
        "queries_per_second": round(nonempty_queries / elapsed, 2) if elapsed else None,
        "similarity": {
            "min": float(sims.min()) if len(sims) else None,
            "median": float(sims.median()) if len(sims) else None,
            "p90": float(sims.quantile(0.90, interpolation="nearest")) if len(sims) else None,
            "max": float(sims.max()) if len(sims) else None,
        },
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--targets-per-source", type=int, default=25_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    result = run_benchmark(
        country=args.country,
        query_limit=args.queries,
        target_limit_per_source=args.targets_per_source,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
    )
    out = Path("artifacts") / "stage4b2_benchmark.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

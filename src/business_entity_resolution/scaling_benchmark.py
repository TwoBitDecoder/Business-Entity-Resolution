"""Stage 4D.4b: characterize target-count scaling without changing retrieval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_one(targets_per_source: int, *, country: str, queries: int,
             chunk_size: int, n_jobs: int) -> dict:
    cmd = [
        sys.executable, "-m",
        "business_entity_resolution.sequential_retrieval_benchmark",
        "--country", country,
        "--queries", str(queries),
        "--targets-per-source", str(targets_per_source),
        "--chunk-size", str(chunk_size),
        "--n-jobs", str(n_jobs),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    report_path = Path("artifacts/stage4d4_phase_memory_diagnostic.json")
    if not report_path.exists():
        raise RuntimeError("child benchmark did not create its report")
    result = json.loads(report_path.read_text(encoding="utf-8"))
    result["targets_per_source_requested"] = targets_per_source
    return result


def run_scaling_benchmark(
    *,
    country: str = "India",
    queries: int = 1_000,
    targets_per_source: tuple[int, ...] = (62_500, 125_000, 250_000),
    chunk_size: int = 250,
    n_jobs: int = -1,
) -> dict:
    if not targets_per_source or any(n < 1 for n in targets_per_source):
        raise ValueError("targets_per_source values must be >= 1")
    runs = [
        _run_one(n, country=country, queries=queries,
                 chunk_size=chunk_size, n_jobs=n_jobs)
        for n in targets_per_source
    ]
    return {
        "purpose": "target-count scaling characterization",
        "retrieval_changed": False,
        "country": country,
        "query_rows_requested": queries,
        "chunk_size": chunk_size,
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", default="62500,125000,250000",
                   help="comma-separated per-source limits; total target pool is 2x")
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    sizes = tuple(int(x.strip()) for x in a.targets_per_source.split(",") if x.strip())
    result = run_scaling_benchmark(
        country=a.country, queries=a.queries,
        targets_per_source=sizes, chunk_size=a.chunk_size, n_jobs=a.n_jobs,
    )
    out = Path("artifacts/stage4d4b_scaling_benchmark.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

from pathlib import Path

import polars as pl

from business_entity_resolution.retrieval_benchmark_runner import run_benchmark


def _write(root: Path, source: str, country: str, rows):
    path = root / source / f"country={country}" / "records.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=["entity_id", "name_compact"], orient="row").write_parquet(path)


def test_runner_is_bounded_and_reports_metrics(tmp_path: Path):
    root = tmp_path / "train"
    _write(root, "source1", "India", [("q1", "acmetrading"), ("q2", "globalfoods")])
    _write(root, "source2", "India", [("a", "acmetradng"), ("b", "other")])
    _write(root, "source3", "India", [("c", "globalfood"), ("d", "different")])

    result = run_benchmark(
        root, country="India", query_limit=2, target_limit_per_source=2, top_k=2, n_jobs=1
    )
    assert result["query_rows"] == 2
    assert result["target_rows"] == 4
    assert result["candidate_rows"] <= result["max_allowed_candidates"] == 4
    assert result["elapsed_seconds"] >= 0
    assert result["similarity"]["max"] is not None

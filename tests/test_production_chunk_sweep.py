import polars as pl

from business_entity_resolution.production_chunk_sweep import run


def _write(root, source, country, rows):
    p = root / source / f"country={country}" / "records.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(p)


def test_chunk_sweep_preserves_topk_bound(tmp_path):
    root = tmp_path / "pre"
    country = "X"
    _write(root, "source1", country, {
        "entity_id": ["q1", "q2", "q3"],
        "name_compact": ["alpha", "beta", "gamma"],
    })
    _write(root, "source2", country, {
        "entity_id": ["a", "b"],
        "name_compact": ["alpha", "beta"],
    })
    _write(root, "source3", country, {
        "entity_id": ["c", "d"],
        "name_compact": ["gamma", "delta"],
    })

    result = run(root, country=country, chunk_sizes=(1, 3))
    assert [r["chunk_size"] for r in result["results"]] == [1, 3]
    for row in result["results"]:
        assert row["candidate_rows"] <= row["chunk_size"] * result["name_top_k"]
        assert row["elapsed_seconds"] >= 0

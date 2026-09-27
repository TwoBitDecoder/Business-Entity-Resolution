from pathlib import Path

import polars as pl

from business_entity_resolution.retrieval_robustness import _sample


def test_random_sample_is_reproducible(tmp_path: Path):
    path = tmp_path / "records.parquet"
    pl.DataFrame({
        "entity_id": [f"q{i}" for i in range(20)],
        "name_compact": [f"name{i}" for i in range(20)],
        "address_norm": [f"address {i}" for i in range(20)],
    }).write_parquet(path)

    a = _sample(path, 7, seed=17)
    b = _sample(path, 7, seed=17)
    c = _sample(path, 7, seed=43)
    assert a["entity_id"].to_list() == b["entity_id"].to_list()
    assert a["entity_id"].to_list() != c["entity_id"].to_list()
    assert a.height == 7


def test_random_sample_caps_at_partition_size(tmp_path: Path):
    path = tmp_path / "records.parquet"
    pl.DataFrame({
        "entity_id": ["a", "b"],
        "name_compact": ["aa", "bb"],
        "address_norm": ["x", "y"],
    }).write_parquet(path)
    assert _sample(path, 10, seed=1).height == 2

from pathlib import Path

import polars as pl
import pytest

from business_entity_resolution.bounded_retrieval import RetrievalConfig
from business_entity_resolution.hybrid_candidates import combine_fuzzy_candidates
from business_entity_resolution.production_candidates import (
    ADDRESS_K,
    FINAL_K,
    NAME_K,
    _load_signal_targets,
    generate_country,
)
from business_entity_resolution.sparse_topk_experiment import sparse_matmul_topk


def _write(root: Path, source: str, country: str, rows):
    path = root / source / f"country={country}" / "records.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _fixture(tmp_path):
    root = tmp_path / "pre"
    country = "X"
    _write(
        root,
        "source1",
        country,
        {
            "entity_id": ["q1", "q2"],
            "name_compact": ["alphaco", "betaco"],
            "address_norm": ["1 main", "2 road"],
        },
    )
    _write(
        root,
        "source2",
        country,
        {
            "entity_id": ["a", "b"],
            "name_compact": ["alphaco", "other"],
            "address_norm": ["1 main", "9 lane"],
        },
    )
    _write(
        root,
        "source3",
        country,
        {
            "entity_id": ["c", "d"],
            "name_compact": ["betaco", "none"],
            "address_norm": ["2 road", "8 street"],
        },
    )
    return root, country


def test_production_generator_chunks_bounds_and_cleanup(tmp_path):
    root, country = _fixture(tmp_path)
    out = tmp_path / "out"

    stats = generate_country(root, out, country, chunk_size=1)

    assert stats["chunks"] == 2
    assert stats["max_candidates_per_query"] <= FINAL_K
    assert stats["name_pass"]["chunks"] == 2
    assert stats["address_pass"]["chunks"] == 2

    parts = sorted((out / f"country={country}").glob("part-*.parquet"))
    assert len(parts) == 2
    got = pl.concat([pl.read_parquet(path) for path in parts])

    assert got.select(["s1_id", "target_id"]).n_unique() == got.height
    assert {"q1", "q2"} <= set(got["s1_id"])
    assert not (out / "_work").exists()


def test_two_pass_output_matches_validated_direct_retrieval(tmp_path):
    root, country = _fixture(tmp_path)
    out = tmp_path / "out"

    generate_country(root, out, country, chunk_size=1)
    actual = pl.concat(
        [
            pl.read_parquet(path)
            for path in sorted((out / f"country={country}").glob("part-*.parquet"))
        ]
    ).sort(["s1_id", "target_id"])

    queries = pl.read_parquet(root / "source1" / f"country={country}" / "records.parquet")
    targets = pl.concat(
        [
            pl.read_parquet(root / source / f"country={country}" / "records.parquet")
            for source in ("source2", "source3")
        ]
    )
    name = sparse_matmul_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=NAME_K),
        text_column="name_compact",
    )
    address = sparse_matmul_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=ADDRESS_K),
        text_column="address_norm",
    )
    expected = combine_fuzzy_candidates(
        name,
        address,
        final_top_k=FINAL_K,
    ).sort(["s1_id", "target_id"])

    assert actual.columns == expected.columns
    assert actual["s1_id"].to_list() == expected["s1_id"].to_list()
    assert actual["target_id"].to_list() == expected["target_id"].to_list()
    assert actual["from_name"].to_list() == expected["from_name"].to_list()
    assert actual["from_address"].to_list() == expected["from_address"].to_list()
    assert actual["name_similarity"].to_list() == expected["name_similarity"].to_list()
    assert actual["address_similarity"].to_list() == expected["address_similarity"].to_list()


def test_signal_target_load_keeps_only_one_text_field(tmp_path):
    root, country = _fixture(tmp_path)

    name_targets = _load_signal_targets(root, country, "name_compact")
    address_targets = _load_signal_targets(root, country, "address_norm")

    assert name_targets.columns == ["entity_id", "name_compact"]
    assert address_targets.columns == ["entity_id", "address_norm"]
    assert name_targets.height == address_targets.height == 4


def test_production_config_is_frozen_r56():
    assert (NAME_K, ADDRESS_K, FINAL_K) == (20, 20, 40)


def test_production_generator_rejects_bad_chunk(tmp_path):
    with pytest.raises(ValueError):
        generate_country(tmp_path, tmp_path / "out", "X", chunk_size=0)

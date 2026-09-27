from pathlib import Path

import polars as pl

from business_entity_resolution.bounded_retrieval import RetrievalConfig
from business_entity_resolution.production_retrieval import (
    SparseTargetIndex,
    iter_parquet_chunks,
    merge_hybrid,
)


def test_reusable_index_queries_multiple_chunks():
    targets = pl.DataFrame({
        "entity_id": ["t1", "t2", "t3"],
        "name_compact": ["acmetrading", "globalfoods", "othercompany"],
    })
    idx = SparseTargetIndex.build(
        targets, config=RetrievalConfig(top_k=1, n_jobs=1),
        text_column="name_compact",
    )
    a = idx.query(pl.DataFrame({"entity_id": ["q1"], "name_compact": ["acmetradng"]}))
    b = idx.query(pl.DataFrame({"entity_id": ["q2"], "name_compact": ["globalfood"]}))
    assert a["target_id"].to_list() == ["t1"]
    assert b["target_id"].to_list() == ["t2"]


def test_parquet_chunks_are_bounded(tmp_path: Path):
    path = tmp_path / "q.parquet"
    pl.DataFrame({
        "entity_id": [f"q{i}" for i in range(7)],
        "name_compact": [f"name{i}" for i in range(7)],
    }).write_parquet(path)
    chunks = list(iter_parquet_chunks(
        path, columns=("entity_id", "name_compact"), chunk_size=3
    ))
    assert [c.height for c in chunks] == [3, 3, 1]
    assert sum(c.height for c in chunks) == 7


def test_merge_hybrid_deduplicates_and_caps():
    name = pl.DataFrame({
        "s1_id": ["q1", "q1"], "target_id": ["t1", "t2"],
        "similarity": [0.9, 0.8],
    })
    address = pl.DataFrame({
        "s1_id": ["q1", "q1"], "target_id": ["t1", "t3"],
        "similarity": [1.0, 0.95],
    })
    result = merge_hybrid(name, address, final_k=2)
    assert result.height == 2
    assert result["target_id"].to_list() == ["t1", "t3"]

import polars as pl
import pytest

from business_entity_resolution.bounded_retrieval import (
    RetrievalConfig,
    retrieve_topk,
    validate_bound,
)


def _frame(rows):
    return pl.DataFrame(rows, schema=["entity_id", "name_compact"], orient="row")


def test_topk_finds_typo_neighbor_and_respects_bound():
    queries = _frame([
        ("q1", "acmetrading"),
        ("q2", "globalfoods"),
    ])
    targets = _frame([
        ("t1", "acmetradng"),
        ("t2", "globalfood"),
        ("t3", "completelydifferent"),
        ("t4", "anothercompany"),
    ])
    cfg = RetrievalConfig(top_k=2, n_jobs=1)
    result = retrieve_topk(queries, targets, config=cfg)

    validate_bound(result, query_count=queries.height, top_k=cfg.top_k)
    assert result.height <= 4
    best = result.sort(["s1_id", "name_similarity"], descending=[False, True]).group_by(
        "s1_id", maintain_order=True
    ).first()
    assert dict(zip(best["s1_id"], best["target_id"])) == {"q1": "t1", "q2": "t2"}


def test_empty_names_do_not_create_candidates():
    queries = _frame([("q1", ""), ("q2", "acme")])
    targets = _frame([("t1", "acme")])
    result = retrieve_topk(queries, targets, config=RetrievalConfig(top_k=5, n_jobs=1))
    assert result["s1_id"].to_list() == ["q2"]
    assert result["target_id"].to_list() == ["t1"]


def test_min_similarity_filters_weak_candidates():
    queries = _frame([("q1", "acmetrading")])
    targets = _frame([("t1", "acmetradng"), ("t2", "zzzzzzzz")])
    result = retrieve_topk(
        queries,
        targets,
        config=RetrievalConfig(top_k=2, min_similarity=0.5, n_jobs=1),
    )
    assert result["target_id"].to_list() == ["t1"]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=0)

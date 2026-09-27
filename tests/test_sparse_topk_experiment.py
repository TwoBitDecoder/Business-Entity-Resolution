import polars as pl

from business_entity_resolution.bounded_retrieval import RetrievalConfig, retrieve_topk
from business_entity_resolution.sparse_topk_experiment import sparse_matmul_topk


def test_sparse_matmul_matches_bruteforce_topk_without_ties():
    queries = pl.DataFrame({
        "entity_id": ["q1", "q2"],
        "name_compact": ["alphatradingcompany", "betatechnologies"],
    })
    targets = pl.DataFrame({
        "entity_id": ["t1", "t2", "t3", "t4"],
        "name_compact": [
            "alphatradingcompany", "alphatradingco",
            "betatechnologies", "unrelatedbakery",
        ],
    })
    cfg = RetrievalConfig(top_k=2, n_jobs=1)

    baseline = retrieve_topk(queries, targets, config=cfg)
    candidate = sparse_matmul_topk(queries, targets, config=cfg)

    baseline_ids = {
        q: set(g["target_id"].to_list())
        for q, g in baseline.group_by("s1_id")
    }
    candidate_ids = {
        q: set(g["target_id"].to_list())
        for q, g in candidate.group_by("s1_id")
    }
    assert candidate_ids == baseline_ids

    baseline_scores = {
        (r["s1_id"], r["target_id"]): r["name_similarity"]
        for r in baseline.iter_rows(named=True)
    }
    for row in candidate.iter_rows(named=True):
        key = (row["s1_id"], row["target_id"])
        assert abs(row["name_similarity"] - baseline_scores[key]) < 1e-5


def test_sparse_matmul_respects_topk():
    queries = pl.DataFrame({"entity_id": ["q"], "name_compact": ["alphacompany"]})
    targets = pl.DataFrame({
        "entity_id": ["a", "b", "c"],
        "name_compact": ["alphacompany", "alphacomp", "alphacorporation"],
    })
    result = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=2, n_jobs=1)
    )
    assert result.height <= 2

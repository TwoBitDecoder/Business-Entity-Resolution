import polars as pl

from business_entity_resolution.r4_address_correctness import compare_address_retrieval


def test_address_sparse_topn_matches_brute_without_ties():
    queries = pl.DataFrame({
        "entity_id": ["q1", "q2"],
        "address_norm": ["12 mg road bengaluru", "44 park street kolkata"],
    })
    targets = pl.DataFrame({
        "entity_id": ["t1", "t2", "t3", "t4"],
        "address_norm": [
            "12 mg road bengaluru", "12 mg rd bengaluru",
            "44 park street kolkata", "99 unrelated avenue",
        ],
    })
    result = compare_address_retrieval(queries, targets, top_k=2)
    assert result["baseline_only_pairs"] == 0
    assert result["sparse_topn_only_pairs"] == 0
    assert result["max_common_score_abs_delta"] < 1e-5


def test_empty_addresses_produce_no_candidates():
    queries = pl.DataFrame({
        "entity_id": ["q-empty", "q-real"],
        "address_norm": ["", "10 market road"],
    })
    targets = pl.DataFrame({
        "entity_id": ["t-empty", "t-real"],
        "address_norm": ["", "10 market road"],
    })
    result = compare_address_retrieval(queries, targets, top_k=10)
    for frame in (result["baseline"], result["sparse_topn"]):
        assert "q-empty" not in frame["s1_id"].to_list()
        assert "t-empty" not in frame["target_id"].to_list()
        assert frame.height == 1
        assert frame["s1_id"][0] == "q-real"
        assert frame["target_id"][0] == "t-real"


def test_address_topk_bound_applies_to_nonempty_queries():
    queries = pl.DataFrame({
        "entity_id": ["q1", "q2"],
        "address_norm": ["10 market road", ""],
    })
    targets = pl.DataFrame({
        "entity_id": ["a", "b", "c"],
        "address_norm": ["10 market road", "10 market rd", "10 market avenue"],
    })
    result = compare_address_retrieval(queries, targets, top_k=2)
    assert result["sparse_topn"].height <= 2

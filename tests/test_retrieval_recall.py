import polars as pl

from business_entity_resolution.retrieval_recall import recall_at_k


def test_recall_at_k_uses_positive_pairs():
    candidates = pl.DataFrame({
        "s1_id": ["q1", "q1", "q1", "q2", "q2"],
        "target_id": ["a", "b", "c", "d", "e"],
        "name_similarity": [0.9, 0.8, 0.7, 0.95, 0.5],
    })
    truth = pl.DataFrame({
        "s1_id": ["q1", "q1", "q2"],
        "target_id": ["a", "c", "e"],
    })
    result = recall_at_k(candidates, truth, ks=(1, 2, 3))
    assert result["recall_at_1"] == 1 / 3
    assert result["recall_at_2"] == 2 / 3
    assert result["recall_at_3"] == 1.0


def test_recall_empty_truth_is_none():
    candidates = pl.DataFrame(
        schema={"s1_id": pl.String, "target_id": pl.String, "name_similarity": pl.Float32}
    )
    truth = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String})
    assert recall_at_k(candidates, truth, ks=(5,)) == {"recall_at_5": None}

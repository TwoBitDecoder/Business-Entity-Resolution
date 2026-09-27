import polars as pl

from business_entity_resolution.r5_hybrid_benchmark import _pair_recall


def test_pair_recall_counts_unique_retrieved_truth_pairs():
    candidates = pl.DataFrame({
        "s1_id": ["q1", "q1", "q2"],
        "target_id": ["a", "x", "b"],
    })
    truth = pl.DataFrame({
        "s1_id": ["q1", "q2", "q2"],
        "target_id": ["a", "b", "c"],
    })
    assert _pair_recall(candidates, truth) == 2 / 3


def test_pair_recall_is_none_without_reachable_truth():
    candidates = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String})
    truth = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String})
    assert _pair_recall(candidates, truth) is None

import polars as pl

from business_entity_resolution.r5_residual_rank_diagnostic import truth_ranks_batched


def test_batched_truth_rank_reports_score_and_rank():
    residuals = pl.DataFrame({
        "s1_id": ["q1"], "target_id": ["truth"], "name_compact": ["alphalimited"]
    })
    targets = pl.DataFrame({
        "entity_id": ["other", "truth", "weak"],
        "name_compact": ["alphalimite", "alphalimited", "zzzz"],
    })
    out = truth_ranks_batched(residuals, targets, text_column="name_compact")[("q1", "truth")]
    assert out["score"] > 0.99
    assert out["rank"] == 1
    assert out["targets_strictly_above"] == 0


def test_batched_truth_rank_handles_empty_signal():
    residuals = pl.DataFrame({
        "s1_id": ["q1"], "target_id": ["truth"], "address_norm": [""]
    })
    targets = pl.DataFrame({"entity_id": ["truth"], "address_norm": [""]})
    out = truth_ranks_batched(residuals, targets, text_column="address_norm")[("q1", "truth")]
    assert out == {"score": 0.0, "rank": None, "targets_strictly_above": None}


def test_batched_truth_rank_handles_multiple_residuals_with_one_target_fit():
    residuals = pl.DataFrame({
        "s1_id": ["q1", "q2"], "target_id": ["a", "b"],
        "name_compact": ["alpha", "beta"],
    })
    targets = pl.DataFrame({
        "entity_id": ["a", "b", "c"], "name_compact": ["alpha", "beta", "gamma"]
    })
    out = truth_ranks_batched(residuals, targets, text_column="name_compact")
    assert out[("q1", "a")]["rank"] == 1
    assert out[("q2", "b")]["rank"] == 1

from business_entity_resolution.r5_residual_rank_diagnostic import truth_rank_for_signal


def test_truth_rank_reports_score_and_rank():
    out = truth_rank_for_signal(
        "alphalimited", "truth", "alphalimited",
        ["other", "truth", "weak"], ["alphalimite", "alphalimited", "zzzz"],
    )
    assert out["score"] > 0.99
    assert out["rank"] == 1
    assert out["targets_strictly_above"] == 0


def test_truth_rank_handles_empty_signal():
    out = truth_rank_for_signal("", "truth", "", ["truth"], [""])
    assert out == {"score": 0.0, "rank": None, "targets_strictly_above": None}

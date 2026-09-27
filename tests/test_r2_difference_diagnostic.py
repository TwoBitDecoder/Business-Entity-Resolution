from business_entity_resolution.r2_difference_diagnostic import _difference_diagnostics


def test_difference_diagnostic_identifies_boundary_tie():
    baseline = {"pairs": [["q", "a", 1.0], ["q", "b", 0.5]]}
    sparse = {"pairs": [["q", "a", 1.0], ["q", "c", 0.5000001]]}
    result = _difference_diagnostics(baseline, sparse)
    assert result["affected_queries"] == 1
    assert result["baseline_only_pairs"] == 1
    assert result["sparse_topn_only_pairs"] == 1
    assert result["baseline_only_near_sparse_boundary"] == 1
    assert result["sparse_only_near_baseline_boundary"] == 1

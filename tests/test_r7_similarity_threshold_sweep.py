from business_entity_resolution.r7_similarity_threshold_sweep import THRESHOLDS

def test_r7_thresholds_are_conservative_and_ordered():
    assert THRESHOLDS[0] == 0.0
    assert list(THRESHOLDS) == sorted(THRESHOLDS)
    assert max(THRESHOLDS) <= 0.20

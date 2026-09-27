from business_entity_resolution.stage7_entity_threshold_sweep import THRESHOLDS, run

def test_threshold_grid_is_bounded_and_complete():
    assert THRESHOLDS[0] == 0.01
    assert THRESHOLDS[-1] == 0.99
    assert len(THRESHOLDS) == 99
    assert callable(run)

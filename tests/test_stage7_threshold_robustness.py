from business_entity_resolution.stage7_threshold_robustness import SEEDS, run

def test_robustness_seeds_are_fixed_and_distinct():
    assert len(SEEDS) == 5
    assert len(set(SEEDS)) == 5
    assert callable(run)

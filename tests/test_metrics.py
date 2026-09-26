from business_entity_resolution.metrics import fbeta_for_entity


def test_singleton_correct():
    assert fbeta_for_entity([], []) == 1.0


def test_singleton_false_merge():
    assert fbeta_for_entity([], ["S2-1"]) == 0.0


def test_precision_heavy_example():
    score = fbeta_for_entity(["S2-47", "S3-812"], ["S2-47", "S2-193", "S3-812"])
    assert round(score, 3) == 0.714

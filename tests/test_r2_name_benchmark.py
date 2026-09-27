import pytest

from business_entity_resolution.r2_name_benchmark import _pair_map


def test_pair_map():
    run = {"pairs": [["q1", "t1", 0.9], ["q1", "t2", 0.8]]}
    assert _pair_map(run) == {("q1", "t1"): 0.9, ("q1", "t2"): 0.8}


def test_pair_map_last_duplicate_wins():
    run = {"pairs": [["q1", "t1", 0.9], ["q1", "t1", 0.8]]}
    assert _pair_map(run)[("q1", "t1")] == pytest.approx(0.8)

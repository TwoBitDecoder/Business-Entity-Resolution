from business_entity_resolution.metrics import fbeta_for_entity


def test_singleton_correct():
    assert fbeta_for_entity([], []) == 1.0


def test_singleton_false_merge():
    assert fbeta_for_entity([], ["S2-1"]) == 0.0


def test_precision_heavy_example():
    score = fbeta_for_entity(["S2-47", "S3-812"], ["S2-47", "S2-193", "S3-812"])
    assert round(score, 3) == 0.714

import polars as pl
import pytest
from business_entity_resolution.metrics import entity_macro_fbeta, fbeta_for_entity, macro_fbeta

def test_singleton_scoring():
    assert fbeta_for_entity([],[]) == 1.0
    assert fbeta_for_entity([],["false"]) == 0.0

def test_fbeta_precision_weighted_value():
    assert fbeta_for_entity(["a","b"],["a"],beta=.5) == pytest.approx(5/6)

def test_entity_macro_includes_entities_with_no_candidate_predictions():
    truth=pl.DataFrame({"s1_id":["q1","q1"],"target_id":["a","b"]})
    scored=pl.DataFrame({
        "s1_id":["q1","q1","q2"],"target_id":["a","x","z"],
        "match_probability":[.9,.1,.1],
    })
    # q1 gets one of two => F0.5=5/6; q2 is true singleton and predicts empty => 1.
    assert entity_macro_fbeta(["q1","q2"],truth,scored,threshold=.5) == pytest.approx((5/6+1)/2)

def test_false_positive_on_singleton_scores_zero():
    truth=pl.DataFrame(schema={"s1_id":pl.String,"target_id":pl.String})
    scored=pl.DataFrame({"s1_id":["q"],"target_id":["x"],"match_probability":[.8]})
    assert entity_macro_fbeta(["q"],truth,scored,threshold=.5) == 0.0

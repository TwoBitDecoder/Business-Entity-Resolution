import polars as pl
import pytest

from business_entity_resolution.features import build_pair_features


def _records():
    q = pl.DataFrame({
        "entity_id": ["q1"],
        "name_norm": ["alpha company"],
        "name_compact": ["alphacompany"],
        "address_norm": ["12 main road 34"],
    })
    t = pl.DataFrame({
        "entity_id": ["t1", "t2"],
        "name_norm": ["alpha company", "beta ltd"],
        "name_compact": ["alphacompany", "betaltd"],
        "address_norm": ["12 main road 99", ""],
    })
    return q, t


def test_pair_features_exact_and_numeric_overlap():
    q, t = _records()
    c = pl.DataFrame({
        "s1_id": ["q1", "q1"],
        "target_id": ["t1", "t2"],
        "name_similarity": [1.0, 0.2],
        "address_similarity": [0.8, 0.0],
        "from_name": [True, True],
        "from_address": [True, False],
    })
    got = build_pair_features(c, q, t)
    exact = got.filter(pl.col("target_id") == "t1").row(0, named=True)
    missing = got.filter(pl.col("target_id") == "t2").row(0, named=True)
    assert exact["name_exact"] is True
    assert exact["name_compact_exact"] is True
    assert exact["address_number_any_overlap"] is True
    assert exact["address_number_jaccard"] == pytest.approx(1 / 3)
    assert missing["target_address_missing"] is True
    assert missing["address_ratio"] == 0.0


def test_pair_features_reject_missing_candidate_id():
    q, t = _records()
    c = pl.DataFrame({
        "s1_id": ["missing"], "target_id": ["t1"],
        "name_similarity": [0.1], "address_similarity": [0.1],
        "from_name": [True], "from_address": [False],
    })
    with pytest.raises(ValueError, match="candidate IDs"):
        build_pair_features(c, q, t)

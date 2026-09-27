import polars as pl
from business_entity_resolution.training_data import (
    add_grouped_split, label_candidate_features, validate_grouped_split,
)

def test_labels_candidate_pairs_and_preserves_negatives():
    f = pl.DataFrame({"s1_id":["q1","q1","q2"],"target_id":["a","b","c"],"x":[1,2,3]})
    truth = pl.DataFrame({"s1_id":["q1"],"target_id":["b"]})
    got = label_candidate_features(f, truth)
    assert got["is_match"].to_list() == [False, True, False]

def test_grouped_split_is_deterministic_and_has_no_s1_leakage():
    f = pl.DataFrame({
        "s1_id": [f"q{i}" for i in range(100) for _ in range(2)],
        "target_id": [f"t{i}" for i in range(200)],
        "is_match": [False] * 200,
    })
    a = add_grouped_split(f, validation_fraction=0.2, seed=42)
    b = add_grouped_split(f, validation_fraction=0.2, seed=42)
    assert a["split"].to_list() == b["split"].to_list()
    validate_grouped_split(a)
    assert set(a["split"].unique().to_list()) == {"train", "validation"}

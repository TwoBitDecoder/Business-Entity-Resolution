import polars as pl
from business_entity_resolution.model import FEATURE_COLUMNS, score_pairs, train_pair_classifier

def _frame(n=40):
    rows=[]
    for i in range(n):
        y=i%4==0
        r={"s1_id":f"q{i//2}","target_id":f"t{i}","is_match":y}
        for j,c in enumerate(FEATURE_COLUMNS):
            r[c] = bool(y) if c in {"from_name","from_address","name_exact","name_compact_exact","address_exact","address_number_any_overlap","query_address_missing","target_address_missing"} else (0.9 if y else 0.1)+(j*0.0001)
        rows.append(r)
    return pl.DataFrame(rows)

def test_lightgbm_baseline_trains_and_scores():
    train=_frame(80)
    valid=_frame(40)
    model=train_pair_classifier(train,valid)
    got=score_pairs(model,valid)
    assert got.height==valid.height
    assert got["match_probability"].min() >= 0
    assert got["match_probability"].max() <= 1

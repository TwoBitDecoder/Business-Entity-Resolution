import polars as pl
from business_entity_resolution.model_artifact import train_and_save,load_model_metadata
from business_entity_resolution.model import FEATURE_COLUMNS

def test_train_and_save_model_artifact(tmp_path):
    rows=[]
    for i in range(40):
        r={"s1_id":f"q{i}","target_id":f"t{i}","split":"train" if i<30 else "validation","is_match":i%2==0}
        r.update({c:float(i%2) for c in FEATURE_COLUMNS})
        rows.append(r)
    meta=train_and_save(pl.DataFrame(rows),tmp_path,seed=42,threshold=.09)
    assert (tmp_path/"pair_model.joblib").is_file()
    assert load_model_metadata(tmp_path)["decision_threshold"]==.09
    assert meta["train_pairs"]==30 and meta["validation_pairs"]==10

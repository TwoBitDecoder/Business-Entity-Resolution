import joblib
import polars as pl
from pathlib import Path
from business_entity_resolution.inference import score_candidate_parts

class DummyModel:
    def predict_proba(self,x):
        import numpy as np
        p=np.full(x.shape[0],.5)
        return np.column_stack([1-p,p])

def test_score_candidate_parts(tmp_path):
    cdir=tmp_path/"c"; cdir.mkdir()
    pl.DataFrame({"s1_id":["q"],"target_id":["t"],"name_similarity":[1.0],"address_similarity":[1.0],"from_name":[True],"from_address":[True]}).write_parquet(cdir/"part-000000.parquet")
    s1=pl.DataFrame({"entity_id":["q"],"name_norm":["alpha"],"name_compact":["alpha"],"address_norm":["1 road"]})
    t=pl.DataFrame({"entity_id":["t"],"name_norm":["alpha"],"name_compact":["alpha"],"address_norm":["1 road"]})
    mp=tmp_path/"m.joblib"; joblib.dump(DummyModel(),mp)
    scored,candidates=score_candidate_parts(mp,cdir,s1,t)
    assert scored.height==1 and candidates.height==1
    assert scored["match_probability"][0]==.5


def test_load_partitioned_records(tmp_path):
    from business_entity_resolution.inference import load_partitioned_records
    for country,eid in [("India","i"),("US","u")]:
        p=tmp_path/"source1"/f"country={country}"; p.mkdir(parents=True)
        pl.DataFrame({"entity_id":[eid]}).write_parquet(p/"records.parquet")
    out=load_partitioned_records(tmp_path,"source1",("India","US"))
    assert set(out["entity_id"])=={"i","u"}

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


def test_production_preflight_rejects_missing_candidate_part(tmp_path):
    import json, pytest
    from business_entity_resolution.inference import validate_production_inputs
    root=tmp_path/"prep"; cand=tmp_path/"cand"; model=tmp_path/"model"; model.mkdir()
    (model/"pair_model.joblib").write_bytes(b"x")
    (model/"metadata.json").write_text(json.dumps({"decision_threshold":.09}))
    for source in ("source1","source2","source3"):
        p=root/source/"country=India"; p.mkdir(parents=True)
        pl.DataFrame({"entity_id":["x"]}).write_parquet(p/"records.parquet")
    p=cand/"country=India"; p.mkdir(parents=True)
    pl.DataFrame({"s1_id":["q"],"target_id":["t"]}).write_parquet(p/"part-000001.parquet")
    with pytest.raises(RuntimeError,match="sequence incomplete"):
        validate_production_inputs(preprocessed_root=root,candidate_root=cand,model_dir=model,countries=("India",))


def test_score_production_parts_resumes_checkpoint(tmp_path):
    from business_entity_resolution.inference import score_production_parts
    prep=tmp_path/"prep"; cand=tmp_path/"cand"; scored=tmp_path/"scored"
    for source in ("source1","source2","source3"):
        p=prep/source/"country=India"; p.mkdir(parents=True)
        pl.DataFrame({"entity_id":["x"],"name_norm":["x"],"name_compact":["x"],"address_norm":["x"]}).write_parquet(p/"records.parquet")
    cp=cand/"country=India"; cp.mkdir(parents=True)
    pl.DataFrame({"s1_id":["x"],"target_id":["x"]}).write_parquet(cp/"part-000000.parquet")
    sp=scored/"country=India"; sp.mkdir(parents=True)
    pl.DataFrame({"s1_id":["x"],"target_id":["x"],"match_probability":[.5]}).write_parquet(sp/"part-000000.parquet")
    model=tmp_path/"model"; model.mkdir(); joblib.dump(DummyModel(),model/"pair_model.joblib")
    stats=score_production_parts(model_dir=model,candidate_root=cand,preprocessed_root=prep,
                                 scored_root=scored,countries=("India",))
    assert stats[0]["resumed_parts"]==1 and stats[0]["written_parts"]==0


def test_assemble_scored_submission_rejects_incomplete_scoring(tmp_path):
    import json, pytest
    from business_entity_resolution.inference import assemble_scored_submission
    prep=tmp_path/"prep"; cand=tmp_path/"cand"; scored=tmp_path/"scored"; model=tmp_path/"model"
    for source in ("source1","source2","source3"):
        p=prep/source/"country=India"; p.mkdir(parents=True)
        pl.DataFrame({"entity_id":["x"]}).write_parquet(p/"records.parquet")
    cp=cand/"country=India"; cp.mkdir(parents=True)
    for i in range(2):
        pl.DataFrame({"s1_id":["x"],"target_id":["x"]}).write_parquet(cp/f"part-{i:06d}.parquet")
    sp=scored/"country=India"; sp.mkdir(parents=True)
    pl.DataFrame({"s1_id":["x"],"target_id":["x"],"match_probability":[.5]}).write_parquet(sp/"part-000000.parquet")
    model.mkdir(); (model/"metadata.json").write_text(json.dumps({"decision_threshold":.09}))
    with pytest.raises(RuntimeError,match="coverage mismatch"):
        assemble_scored_submission(preprocessed_root=prep,candidate_root=cand,scored_root=scored,
                                   model_dir=model,output_dir=tmp_path/"out",countries=("India",))

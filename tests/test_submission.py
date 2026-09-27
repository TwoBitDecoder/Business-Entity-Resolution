import polars as pl
import pytest
from business_entity_resolution.submission import build_matching_results,build_candidate_pairs,validate_submission

def test_submission_build_and_validate():
    s1=pl.DataFrame({"entity_id":["q1","q2"]})
    targets=pl.DataFrame({"entity_id":["a","b"]})
    cand=pl.DataFrame({"s1_id":["q1","q1","q2"],"target_id":["a","b","b"]})
    scored=cand.with_columns(pl.Series("match_probability",[.9,.01,.08]))
    m=build_matching_results(s1,scored,threshold=.09)
    p=build_candidate_pairs(cand)
    validate_submission(s1,targets,m,p)
    assert m.filter(pl.col("source1_entity_id")=="q1")["matched_entity_ids"][0]=="a"
    assert m.filter(pl.col("source1_entity_id")=="q2")["matched_entity_ids"][0]==""

def test_submission_rejects_match_outside_candidates():
    s1=pl.DataFrame({"entity_id":["q1"]}); targets=pl.DataFrame({"entity_id":["a"]})
    m=pl.DataFrame({"source1_entity_id":["q1"],"matched_entity_ids":["a"]})
    p=pl.DataFrame(schema={"source1_entity_id":pl.String,"candidate_entity_id":pl.String})
    with pytest.raises(ValueError,match="missing from candidate_pairs"):
        validate_submission(s1,targets,m,p)

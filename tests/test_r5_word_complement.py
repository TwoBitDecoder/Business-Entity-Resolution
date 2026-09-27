import polars as pl
from business_entity_resolution.r5_word_complement import word_topk

def test_word_topk_is_bounded_and_finds_reordered_tokens():
    q=pl.DataFrame({"entity_id":["q1"],"name_norm":["alpha trading company"]})
    t=pl.DataFrame({"entity_id":["x","truth","y"],"name_norm":["omega services","company alpha trading","beta foods"]})
    out=word_topk(q,t,top_k=2)
    assert out.height <= 2
    assert "truth" in out["target_id"].to_list()
    assert out.columns == ["s1_id","target_id","word_similarity"]

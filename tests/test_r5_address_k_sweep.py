import polars as pl
from business_entity_resolution.hybrid_candidates import combine_fuzzy_candidates

def test_address_k_sweep_budget_logic():
    name=pl.DataFrame({"s1_id":["q"]*2,"target_id":["n1","n2"],"name_similarity":[.9,.8]})
    address=pl.DataFrame({"s1_id":["q"]*3,"target_id":["a1","a2","a3"],"name_similarity":[.95,.7,.6]})
    out=combine_fuzzy_candidates(name,address,final_top_k=5)
    assert out.height == 5
    assert out.group_by("s1_id").len()["len"].max() <= 5

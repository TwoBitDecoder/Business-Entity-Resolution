import polars as pl
from business_entity_resolution.bounded_retrieval import RetrievalConfig
from business_entity_resolution.reusable_sparse_index import SparseTopKIndex
from business_entity_resolution.sparse_topk_experiment import sparse_matmul_topk

def test_reusable_index_matches_validated_primitive():
    t=pl.DataFrame({"entity_id":["a","b","c"],"name_compact":["alpha","beta","gamma"]})
    q=pl.DataFrame({"entity_id":["q1","q2"],"name_compact":["alph","bet"]})
    cfg=RetrievalConfig(top_k=2)
    expected=sparse_matmul_topk(q,t,config=cfg,text_column="name_compact").sort(["s1_id","target_id"])
    got=SparseTopKIndex(t,text_column="name_compact",config=cfg).query(q).sort(["s1_id","target_id"])
    assert got["s1_id"].to_list()==expected["s1_id"].to_list()
    assert got["target_id"].to_list()==expected["target_id"].to_list()
    assert got["name_similarity"].to_list()==expected["name_similarity"].to_list()

def test_reusable_index_can_query_multiple_chunks():
    t=pl.DataFrame({"entity_id":["a","b"],"name_compact":["alpha","beta"]})
    idx=SparseTopKIndex(t,text_column="name_compact",config=RetrievalConfig(top_k=1))
    for qid,text in [("q1","alpha"),("q2","beta")]:
        got=idx.query(pl.DataFrame({"entity_id":[qid],"name_compact":[text]}))
        assert got.height==1
        assert got["s1_id"][0]==qid

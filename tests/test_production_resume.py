from pathlib import Path
import polars as pl
from business_entity_resolution import production_candidates as pc

class FakeIndex:
    calls=0
    def __init__(self,*a,**k): pass
    def query(self,q):
        FakeIndex.calls+=1
        return pl.DataFrame({"s1_id":q["entity_id"],"target_id":["t"]*q.height,"similarity":[1.0]*q.height})

def test_signal_pass_resumes_existing_part(tmp_path,monkeypatch):
    qpath=tmp_path/"q.parquet"
    pl.DataFrame({"entity_id":["q1","q2"],"name_compact":["a","b"]}).write_parquet(qpath)
    monkeypatch.setattr(pc,"_load_signal_targets",lambda *a,**k: pl.DataFrame({"entity_id":["t"],"name_compact":["a"]}))
    monkeypatch.setattr(pc,"SparseTopKIndex",FakeIndex)
    work=tmp_path/"work"; work.mkdir()
    pl.DataFrame({"s1_id":["q1"],"target_id":["t"],"similarity":[1.0]}).write_parquet(work/"part-000000.parquet")
    FakeIndex.calls=0
    stats=pc._run_signal_pass(root=tmp_path,qpath=qpath,work_dir=work,country="X",
        text_column="name_compact",label="name",top_k=20,qrows=2,chunk_size=1)
    assert stats["resumed_chunks"]==1
    assert FakeIndex.calls==1
    assert stats["chunks"]==2

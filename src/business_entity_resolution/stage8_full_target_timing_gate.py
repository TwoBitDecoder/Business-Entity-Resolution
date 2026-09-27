"""Stage 8 gate: full India target pool with a small query sample.

Measures the actual production target-index scale without launching a full
country query pass. Frozen char 3-5, Top20, 2 threads.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import polars as pl
from .bounded_retrieval import RetrievalConfig
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex

QUERY_COUNT=1000

def run(root="artifacts/preprocessed/train",country="India")->dict:
    root=Path(root)
    qpath=_partition_path(root,"source1",country)
    queries=(pl.scan_parquet(qpath).select("entity_id","name_compact")
             .slice(0,QUERY_COUNT).collect(engine="streaming"))
    targets=pl.concat([
        pl.scan_parquet(_partition_path(root,s,country))
          .select("entity_id","name_compact").collect(engine="streaming")
        for s in ("source2","source3")
    ])
    started=time.perf_counter()
    idx=SparseTopKIndex(targets,text_column="name_compact",
                        config=RetrievalConfig(top_k=20,n_jobs=2))
    index_seconds=time.perf_counter()-started
    started=time.perf_counter()
    candidates=idx.query(queries)
    query_seconds=time.perf_counter()-started
    qps=queries.height/query_seconds
    full_queries=(pl.scan_parquet(qpath).select(pl.len())
                  .collect(engine="streaming").item())
    result={"country":country,"threads":2,"sample_queries":queries.height,
            "full_target_rows":targets.height,"candidate_rows":candidates.height,
            "index_seconds":index_seconds,"query_seconds":query_seconds,
            "queries_per_second":qps,"full_country_queries":full_queries,
            "projected_full_name_query_hours":full_queries/qps/3600}
    out=Path("artifacts/stage8_full_target_timing_gate.json")
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    run()

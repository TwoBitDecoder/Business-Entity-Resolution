"""Stage 8 production-path timing gate after validated 2-thread tuning.

Runs the actual reusable production index/query path on a bounded sample and
projects query time to the full country. It does not launch full retrieval.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import polars as pl
from .bounded_retrieval import RetrievalConfig
from .production_candidates import _load_signal_targets
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex

QUERY_COUNT=5000
TARGET_LIMIT_PER_SOURCE=250000

def run(root="artifacts/preprocessed/train",country="India")->dict:
    root=Path(root)
    qpath=_partition_path(root,"source1",country)
    queries=(pl.scan_parquet(qpath).select("entity_id","name_compact")
             .slice(0,QUERY_COUNT).collect(engine="streaming"))
    # Keep this gate bounded to the same 500K target scale used by prior timing
    # experiments while exercising the production index/query implementation.
    frames=[]
    for source in ("source2","source3"):
        frames.append(pl.scan_parquet(_partition_path(root,source,country))
                      .select("entity_id","name_compact")
                      .slice(0,TARGET_LIMIT_PER_SOURCE).collect(engine="streaming"))
    targets=pl.concat(frames)
    started=time.perf_counter()
    idx=SparseTopKIndex(targets,text_column="name_compact",
                        config=RetrievalConfig(top_k=20,n_jobs=2))
    index_seconds=time.perf_counter()-started
    started=time.perf_counter(); candidates=idx.query(queries)
    query_seconds=time.perf_counter()-started
    qps=queries.height/query_seconds
    full_queries=(pl.scan_parquet(qpath).select(pl.len())
                  .collect(engine="streaming").item())
    result={"country":country,"threads":2,"sample_queries":queries.height,
            "sample_targets":targets.height,"candidate_rows":candidates.height,
            "index_seconds":index_seconds,"query_seconds":query_seconds,
            "queries_per_second":qps,"full_country_queries":full_queries,
            "projected_name_query_hours_at_sample_scale":full_queries/qps/3600}
    out=Path("artifacts/stage8_production_timing_gate.json")
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    run()

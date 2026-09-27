"""Stage 8 benchmark: sparse-dot-topn thread scaling, frozen retrieval semantics.

Changes only n_threads. Representation, target index, Top-K and query batch are fixed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex

QUERY_COUNT=5000
TARGET_LIMIT_PER_SOURCE=250000


def _signature(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("s1_id","target_id").sort(["s1_id","target_id"])


def run(root="artifacts/preprocessed/train",country="India")->dict:
    root=Path(root)
    queries=_load(_partition_path(root,"source1",country),QUERY_COUNT)
    targets=pl.concat([_load(_partition_path(root,s,country),TARGET_LIMIT_PER_SOURCE)
                       for s in ("source2","source3")])
    cpu=os.cpu_count() or 1
    thread_counts=tuple(dict.fromkeys([1,2,4,8,cpu]))
    runs=[]; baseline=None
    for threads in thread_counts:
        idx=SparseTopKIndex(targets,text_column="name_compact",
                            config=RetrievalConfig(top_k=20,n_jobs=threads))
        started=time.perf_counter()
        result=idx.query(queries)
        elapsed=time.perf_counter()-started
        sig=_signature(result)
        identical=True if baseline is None else sig.equals(baseline)
        if baseline is None: baseline=sig
        runs.append({"threads":threads,"seconds":elapsed,
                     "queries_per_second":queries.height/elapsed,
                     "candidate_rows":result.height,
                     "identical_to_single_thread":identical})
        print(f"threads={threads}: {elapsed:.2f}s, {queries.height/elapsed:.1f} q/s, identical={identical}",flush=True)
    single=runs[0]["seconds"]
    for item in runs: item["speedup_vs_single_thread"]=single/item["seconds"]
    out={"country":country,"cpu_count":cpu,"query_count":queries.height,
         "target_rows":targets.height,"runs":runs}
    path=Path("artifacts/stage8_thread_scaling_benchmark.json")
    path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2)); return out


if __name__=="__main__":
    run()

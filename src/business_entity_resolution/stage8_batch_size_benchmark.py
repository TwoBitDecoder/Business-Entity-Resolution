"""Stage 8 benchmark: same frozen char retrieval, larger query batches.

Production currently queries 1,000 S1 rows at a time. sparse-dot-topn pays
substantial fixed work per call, so this benchmark changes only query batch size.
The fitted target representation and exact Top-K algorithm stay unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex

BATCH_SIZES=(1000,5000,10000)
QUERY_COUNT=10000
TARGET_LIMIT_PER_SOURCE=250000


def _signature(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("s1_id","target_id").sort(["s1_id","target_id"])


def run(root="artifacts/preprocessed/train",country="India")->dict:
    root=Path(root)
    queries=_load(_partition_path(root,"source1",country),QUERY_COUNT)
    targets=pl.concat([
        _load(_partition_path(root,s,country),TARGET_LIMIT_PER_SOURCE)
        for s in ("source2","source3")
    ])
    started=time.perf_counter()
    index=SparseTopKIndex(targets,text_column="name_compact",
                          config=RetrievalConfig(top_k=20))
    index_seconds=time.perf_counter()-started

    runs=[]; baseline=None
    for batch_size in BATCH_SIZES:
        parts=[]; started=time.perf_counter()
        for offset in range(0,queries.height,batch_size):
            parts.append(index.query(queries.slice(offset,batch_size)))
        elapsed=time.perf_counter()-started
        result=pl.concat(parts)
        sig=_signature(result)
        if baseline is None:
            baseline=sig
            identical=True
        else:
            identical=sig.equals(baseline)
        runs.append({"batch_size":batch_size,"seconds":elapsed,
                     "queries_per_second":queries.height/elapsed,
                     "candidate_rows":result.height,
                     "identical_to_1000":identical})
        print(f"batch={batch_size}: {elapsed:.2f}s, {queries.height/elapsed:.1f} q/s, identical={identical}",flush=True)

    base=runs[0]["seconds"]
    for item in runs:
        item["speedup_vs_1000"]=base/item["seconds"]

    out={"country":country,"query_count":queries.height,"target_rows":targets.height,
         "index_seconds":index_seconds,"runs":runs}
    path=Path("artifacts/stage8_batch_size_benchmark.json")
    path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2)); return out


if __name__=="__main__":
    run()

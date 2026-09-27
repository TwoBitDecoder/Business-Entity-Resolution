"""Stage 8 benchmark: exact sparse retrieval with a word-boundary name index.

This does not replace production retrieval. It tests one controlled speed/quality
change against the frozen char 3-5 gram name retriever on the same bounded data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .bounded_retrieval import RetrievalConfig
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk

K=20


def _word_topk(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    q=queries.select("entity_id","name_norm").filter(pl.col("name_norm")!="")
    t=targets.select("entity_id","name_norm").filter(pl.col("name_norm")!="")
    v=TfidfVectorizer(analyzer="word",ngram_range=(1,2),lowercase=False,
                      dtype=np.float32,norm="l2",sublinear_tf=True)
    tm=v.fit_transform(t["name_norm"].to_list()).tocsr()
    qm=v.transform(q["name_norm"].to_list()).tocsr()
    sims=sp_matmul_topn(qm,tm.T.tocsc(),top_n=min(K,t.height),
                        threshold=0.0,sort=True).tocsr()
    qids=q["entity_id"].to_list(); tids=t["entity_id"].to_list(); rows=[]
    for i,qid in enumerate(qids):
        for j,s in zip(sims.indices[sims.indptr[i]:sims.indptr[i+1]],
                       sims.data[sims.indptr[i]:sims.indptr[i+1]],strict=True):
            rows.append((qid,tids[int(j)],float(s)))
    return pl.DataFrame(rows,schema=["s1_id","target_id","name_similarity"],
                        orient="row").with_columns(pl.col("name_similarity").cast(pl.Float32))


def _recall(candidates: pl.DataFrame, truth_pool: pl.DataFrame) -> tuple[int,int,float]:
    got=truth_pool.join(candidates.select("s1_id","target_id").unique(),
                        on=["s1_id","target_id"],how="inner").height
    total=truth_pool.height
    return got,total,(got/total if total else 0.0)


def run(root="artifacts/preprocessed/train",country="India",query_count=1000,
        target_limit_per_source=250000)->dict:
    root=Path(root)
    q=_load(_partition_path(root,"source1",country),query_count)
    t=pl.concat([_load(_partition_path(root,s,country),target_limit_per_source)
                 for s in ("source2","source3")])
    truth=load_truth("data/train/train_ground_truth.tsv",q["entity_id"].to_list())
    target_ids=t.select(pl.col("entity_id").alias("target_id")).unique()
    truth_pool=truth.join(target_ids,on="target_id",how="inner")

    start=time.perf_counter()
    frozen=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=K),text_column="name_compact")
    frozen_s=time.perf_counter()-start
    start=time.perf_counter(); word=_word_topk(q,t); word_s=time.perf_counter()-start
    fg,ft,fr=_recall(frozen,truth_pool); wg,wt,wr=_recall(word,truth_pool)
    result={"country":country,"query_count":query_count,"target_rows":t.height,
            "frozen_char":{"seconds":frozen_s,"candidate_rows":frozen.height,
                           "truth_pool":ft,"retrieved_truth":fg,"recall":fr},
            "word_1_2":{"seconds":word_s,"candidate_rows":word.height,
                        "truth_pool":wt,"retrieved_truth":wg,"recall":wr},
            "speedup":frozen_s/word_s if word_s else None,
            "recall_delta":wr-fr}
    out=Path("artifacts/stage8_name_retrieval_benchmark.json")
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); return result


if __name__=="__main__":
    run()

"""Deadline fallback benchmark: exact-name rescue + fast word retrieval.

This is experiment-only code. It does not modify the validated production retriever.
Run on the same bounded India universe before any promotion.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .retrieval_benchmark_runner import _partition_path


def _load(path: Path, limit: int, cols: list[str]) -> pl.DataFrame:
    return pl.scan_parquet(path).select(cols).head(limit).collect(engine="streaming")


def _word_topk(queries: pl.DataFrame, targets: pl.DataFrame, column: str, top_k: int) -> pl.DataFrame:
    t=targets.select("entity_id",column).filter(pl.col(column)!="")
    q=queries.select("entity_id",column).filter(pl.col(column)!="")
    vec=TfidfVectorizer(analyzer="word",ngram_range=(1,2),lowercase=False,dtype=np.float32,norm="l2")
    tm=vec.fit_transform(t[column].to_list()).tocsr()
    qm=vec.transform(q[column].to_list()).tocsr()
    sims=sp_matmul_topn(qm,tm.T.tocsc(),top_n=top_k,threshold=0.0,sort=True,n_threads=2).tocsr()
    tids=t["entity_id"].to_list(); qids=q["entity_id"].to_list(); rows=[]
    for i,qid in enumerate(qids):
        for j in range(sims.indptr[i],sims.indptr[i+1]):
            rows.append((qid,tids[int(sims.indices[j])],float(sims.data[j])))
    return pl.DataFrame(rows,schema=["s1_id","target_id","score"],orient="row")


def _exact_name(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    q=queries.select(pl.col("entity_id").alias("s1_id"),"name_compact").filter(pl.col("name_compact")!="")
    t=targets.select(pl.col("entity_id").alias("target_id"),"name_compact").filter(pl.col("name_compact")!="")
    return q.join(t,on="name_compact",how="inner").select("s1_id","target_id")


def run(*,root="artifacts/preprocessed/train",country="India",queries=1000,
        targets_per_source=250000,top_k=40,ground_truth="data/train/train_ground_truth.tsv"):
    base=Path(root); cols=["entity_id","name_compact"]
    q=_load(_partition_path(base,"source1",country),queries,cols)
    s2=_load(_partition_path(base,"source2",country),targets_per_source,cols)
    s3=_load(_partition_path(base,"source3",country),targets_per_source,cols)
    targets=pl.concat([s2,s3],how="vertical")
    target_ids=set(targets["entity_id"].to_list()); qids=set(q["entity_id"].to_list())
    gt=pl.read_csv(ground_truth,separator="\t",infer_schema=False)
    truth=set()
    for row in gt.filter(pl.col("source1_entity_id").is_in(qids)).iter_rows(named=True):
        raw=row.get("matched_entity_ids") or ""
        for tid in raw.split(","):
            tid=tid.strip()
            if tid and tid in target_ids: truth.add((row["source1_entity_id"],tid))
    started=time.perf_counter()
    word_name=_word_topk(q,targets,"name_compact",top_k)
    # Address word retrieval is the critical second signal from the validated hybrid baseline.
    qaddr=_load(_partition_path(base,"source1",country),queries,["entity_id","address_norm"])
    s2addr=_load(_partition_path(base,"source2",country),targets_per_source,["entity_id","address_norm"])
    s3addr=_load(_partition_path(base,"source3",country),targets_per_source,["entity_id","address_norm"])
    taddr=pl.concat([s2addr,s3addr],how="vertical")
    word_addr=_word_topk(qaddr,taddr,"address_norm",top_k)
    exact=_exact_name(q,targets)
    candidates=pl.concat([word_name.select("s1_id","target_id"),word_addr.select("s1_id","target_id"),exact],how="vertical").unique()
    elapsed=time.perf_counter()-started
    got=set(candidates.iter_rows())
    return {"query_rows":q.height,"target_rows":targets.height,"truth_pool":len(truth),
            "retrieved_truth":len(truth & got),"recall":len(truth & got)/len(truth) if truth else None,
            "candidate_rows":candidates.height,"elapsed_seconds":round(elapsed,3),
            "queries_per_second":round(q.height/elapsed,2) if elapsed else None,
            "config":{"word_ngram":[1,2],"name_top_k":top_k,"address_top_k":top_k,"exact_name_rescue":True}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--top-k",type=int,default=40)
    p.add_argument("--targets-per-source",type=int,default=250000)
    a=p.parse_args(); result=run(top_k=a.top_k,targets_per_source=a.targets_per_source)
    print(json.dumps(result,indent=2))
    out=Path("artifacts/stage8c_deadline_fallback.json"); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()

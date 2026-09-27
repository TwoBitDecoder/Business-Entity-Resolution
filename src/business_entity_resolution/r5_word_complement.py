"""R5.5: bounded word-token TF-IDF complement experiment.

Tests whether word n-grams add truth pairs missed by the frozen char-name +
char-address hybrid. Experimental only; production retrieval is unchanged.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .r5_hybrid_benchmark import _load, _pair_recall, _max_rss_mb


def word_topk(queries: pl.DataFrame, targets: pl.DataFrame, *, top_k: int = 10) -> pl.DataFrame:
    q=queries.select("entity_id","name_norm").filter(pl.col("name_norm")!="")
    t=targets.select("entity_id","name_norm").filter(pl.col("name_norm")!="")
    if q.is_empty() or t.is_empty():
        return pl.DataFrame(schema={"s1_id":pl.String,"target_id":pl.String,"word_similarity":pl.Float32})
    v=TfidfVectorizer(analyzer="word",ngram_range=(1,2),lowercase=False,dtype=np.float32,norm="l2",
                      sublinear_tf=True)
    tm=v.fit_transform(t["name_norm"].to_list()).tocsr()
    qm=v.transform(q["name_norm"].to_list()).tocsr()
    sims=sp_matmul_topn(qm,tm.T.tocsc(),top_n=min(top_k,t.height),threshold=0.0,sort=True).tocsr()
    qids=q["entity_id"].to_list(); tids=t["entity_id"].to_list(); rows=[]
    for qi,qid in enumerate(qids):
        s,e=sims.indptr[qi],sims.indptr[qi+1]
        for j in range(s,e):
            rows.append((qid,tids[int(sims.indices[j])],float(sims.data[j])))
    return pl.DataFrame(rows,schema=["s1_id","target_id","word_similarity"],orient="row").with_columns(
        pl.col("word_similarity").cast(pl.Float32))


def run(country="India", query_count=1000, target_limit_per_source=250000, word_k=10):
    base=Path("artifacts/preprocessed/train"); started=time.perf_counter()
    cols=["entity_id","name_norm","name_compact","address_norm"]
    def load(source,limit):
        return pl.scan_parquet(_partition_path(base,source,country)).select(cols).head(limit).collect(engine="streaming")
    q=load("source1",query_count)
    t=pl.concat([load("source2",target_limit_per_source),load("source3",target_limit_per_source)])
    name=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=20),text_column="name_compact")
    address=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=10),text_column="address_norm")
    hybrid=combine_fuzzy_candidates(name,address,final_top_k=30)
    ts=time.perf_counter(); word=word_topk(q,t,top_k=word_k); word_s=time.perf_counter()-ts
    union=pl.concat([
        hybrid.select("s1_id","target_id"),
        word.select("s1_id","target_id")
    ]).unique()
    truth_all=load_truth("data/train/train_ground_truth.tsv",q["entity_id"].to_list())
    truth_pool=truth_all.join(t.select(pl.col("entity_id").alias("target_id")).unique(),on="target_id",how="inner")
    hybrid_hits=hybrid.select("s1_id","target_id").join(truth_pool,on=["s1_id","target_id"],how="inner").unique()
    union_hits=union.join(truth_pool,on=["s1_id","target_id"],how="inner").unique()
    return {"purpose":"R5.5 bounded word-token complement experiment","country":country,
      "query_rows":q.height,"target_rows":t.height,"word_top_k":word_k,
      "candidate_rows":{"hybrid":hybrid.height,"word":word.height,"union":union.height},
      "truth_pairs_in_target_pool":truth_pool.height,
      "truth_pairs_retrieved":{"hybrid":hybrid_hits.height,"hybrid_plus_word":union_hits.height,
        "incremental":union_hits.height-hybrid_hits.height},
      "pair_recall":{"hybrid":_pair_recall(hybrid,truth_pool),"hybrid_plus_word":_pair_recall(union,truth_pool)},
      "word_retrieval_seconds":word_s,"total_seconds":time.perf_counter()-started,
      "process_max_rss_mb":_max_rss_mb()}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--word-k",type=int,default=10)
    a=p.parse_args(); result=run(word_k=a.word_k)
    out=Path("artifacts/retrieval_v2_r5_word_complement.json")
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved: {out}")
if __name__=="__main__": main()

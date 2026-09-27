"""R5.6: address Top-K recall/cost sweep on the frozen sparse signal."""
from __future__ import annotations
import json, time
from pathlib import Path
import polars as pl
from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .r5_hybrid_benchmark import _load, _pair_recall, _max_rss_mb


def run(country="India", query_count=1000, target_limit_per_source=250000,
        address_ks=(10,15,20,30)):
    base=Path("artifacts/preprocessed/train"); started=time.perf_counter()
    q=_load(_partition_path(base,"source1",country),query_count)
    t=pl.concat([_load(_partition_path(base,"source2",country),target_limit_per_source),
                 _load(_partition_path(base,"source3",country),target_limit_per_source)])
    name=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=20),text_column="name_compact")
    # Retrieve address Top-30 once. Lower-K experiments are deterministic prefixes,
    # avoiding four expensive 500K-target TF-IDF rebuilds.
    ts=time.perf_counter()
    addr_max=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=max(address_ks)),
                                text_column="address_norm")
    address_seconds=time.perf_counter()-ts
    addr_ranked=addr_max.with_columns(
        pl.int_range(pl.len()).over("s1_id").alias("_rank0")
    )
    truth_all=load_truth("data/train/train_ground_truth.tsv",q["entity_id"].to_list())
    truth_pool=truth_all.join(t.select(pl.col("entity_id").alias("target_id")).unique(),
                              on="target_id",how="inner")
    results=[]
    for k in address_ks:
        address=addr_ranked.filter(pl.col("_rank0")<k).drop("_rank0")
        # Preserve the existing total budget policy: name20 + addressK.
        hybrid=combine_fuzzy_candidates(name,address,final_top_k=20+k)
        hits=round((_pair_recall(hybrid,truth_pool) or 0)*truth_pool.height)
        results.append({"address_top_k":k,"final_top_k":20+k,
                        "candidate_rows":hybrid.height,"truth_pairs_retrieved":hits,
                        "pair_recall":_pair_recall(hybrid,truth_pool)})
    return {"purpose":"R5.6 address Top-K recall/cost sweep","country":country,
            "query_rows":q.height,"target_rows":t.height,
            "truth_pairs_in_target_pool":truth_pool.height,
            "name_top_k":20,"address_top_k_max_computed":max(address_ks),
            "results":results,"address_top30_retrieval_seconds":address_seconds,
            "total_seconds":time.perf_counter()-started,
            "process_max_rss_mb":_max_rss_mb()}


def main():
    result=run()
    out=Path("artifacts/retrieval_v2_r5_address_k_sweep.json")
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved: {out}")
if __name__=="__main__": main()

"""Train and persist the frozen bounded LightGBM model for deadline submission."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import polars as pl
from .bounded_retrieval import RetrievalConfig
from .features import build_pair_features
from .hybrid_candidates import combine_fuzzy_candidates
from .model_artifact import train_and_save
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .training_data import add_grouped_split, label_candidate_features

def run(root="artifacts/preprocessed/train", output="artifacts/model",
        country="India", query_count=1000, target_limit_per_source=250000):
    started=time.perf_counter(); root=Path(root)
    q=_load(_partition_path(root,"source1",country),query_count)
    targets=pl.concat([_load(_partition_path(root,s,country),target_limit_per_source)
                       for s in ("source2","source3")],how="vertical")
    name=sparse_matmul_topk(q,targets,config=RetrievalConfig(top_k=20,n_jobs=2),
                            text_column="name_compact")
    addr=sparse_matmul_topk(q,targets,config=RetrievalConfig(top_k=20,n_jobs=2),
                            text_column="address_norm")
    cand=combine_fuzzy_candidates(name,addr,final_top_k=40)
    feat=build_pair_features(cand,q,targets)
    truth=load_truth("data/train/train_ground_truth.tsv",q["entity_id"].to_list())
    labelled=add_grouped_split(label_candidate_features(feat,truth),seed=42)
    positives=int(labelled["is_match"].sum())
    if positives==0: raise RuntimeError("bounded training sample contains no positive pairs")
    meta=train_and_save(labelled,output,seed=42,threshold=.09)
    result={**meta,"country":country,"query_rows":q.height,"target_rows":targets.height,
            "candidate_pairs":cand.height,"positive_pairs":positives,
            "elapsed_seconds":round(time.perf_counter()-started,3)}
    print(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="artifacts/preprocessed/train")
    p.add_argument("--output",default="artifacts/model")
    p.add_argument("--country",default="India")
    p.add_argument("--query-count",type=int,default=1000)
    p.add_argument("--target-limit-per-source",type=int,default=250000)
    a=p.parse_args(); run(a.root,a.output,a.country,a.query_count,a.target_limit_per_source)

if __name__=="__main__": main()

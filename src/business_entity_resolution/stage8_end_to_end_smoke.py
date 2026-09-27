"""Bounded end-to-end smoke test: retrieval -> features -> model -> inference -> submission."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import polars as pl
from .bounded_retrieval import RetrievalConfig
from .features import build_pair_features
from .hybrid_candidates import combine_fuzzy_candidates
from .model import train_pair_classifier
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .submission import build_candidate_pairs,build_matching_results,validate_submission
from .training_data import add_grouped_split,label_candidate_features

def run(root="artifacts/preprocessed/train",country="India",query_count=200,target_limit_per_source=50000):
    root=Path(root); q=_load(_partition_path(root,"source1",country),query_count)
    t=pl.concat([_load(_partition_path(root,s,country),target_limit_per_source) for s in ("source2","source3")])
    name=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=20,n_jobs=2),text_column="name_compact")
    addr=sparse_matmul_topk(q,t,config=RetrievalConfig(top_k=20,n_jobs=2),text_column="address_norm")
    cand=combine_fuzzy_candidates(name,addr,final_top_k=40)
    feat=build_pair_features(cand,q,t)
    truth=load_truth("data/train/train_ground_truth.tsv",q["entity_id"].to_list())
    labelled=add_grouped_split(label_candidate_features(feat,truth),seed=42)
    train=labelled.filter(pl.col("split")=="train"); valid=labelled.filter(pl.col("split")=="validation")
    model=train_pair_classifier(train,valid,seed=42)
    from .model import score_pairs
    scored=score_pairs(model,feat)
    matching=build_matching_results(q,scored,threshold=.09); pairs=build_candidate_pairs(cand)
    validate_submission(q,t,matching,pairs)
    result={"query_rows":q.height,"target_rows":t.height,"candidate_pairs":pairs.height,
            "matching_rows":matching.height,"train_pairs":train.height,"validation_pairs":valid.height}
    out=Path("artifacts/stage8_end_to_end_smoke.json"); out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); return result
if __name__=="__main__": run()

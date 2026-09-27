"""Stage 7 bounded real-data LightGBM baseline audit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .bounded_retrieval import RetrievalConfig
from .features import build_pair_features
from .hybrid_candidates import combine_fuzzy_candidates
from .model import score_pairs, train_pair_classifier
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .training_data import add_grouped_split, label_candidate_features


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1000,
    target_limit_per_source: int = 250000,
) -> dict:
    base=Path(root)
    queries=_load(_partition_path(base,"source1",country),query_count)
    targets=pl.concat([
        _load(_partition_path(base,"source2",country),target_limit_per_source),
        _load(_partition_path(base,"source3",country),target_limit_per_source),
    ])
    print(f"{country}: retrieving candidates...",flush=True)
    name=sparse_matmul_topk(queries,targets,config=RetrievalConfig(top_k=20),text_column="name_compact")
    address=sparse_matmul_topk(queries,targets,config=RetrievalConfig(top_k=20),text_column="address_norm")
    candidates=combine_fuzzy_candidates(name,address,final_top_k=40)

    print(f"{country}: building labelled feature data...",flush=True)
    features=build_pair_features(candidates,queries,targets)
    truth=load_truth("data/train/train_ground_truth.tsv",queries["entity_id"].to_list())
    data=add_grouped_split(label_candidate_features(features,truth))
    train=data.filter(pl.col("split")=="train")
    valid=data.filter(pl.col("split")=="validation")

    print(f"{country}: training LightGBM...",flush=True)
    t0=time.perf_counter()
    model=train_pair_classifier(train,valid)
    train_seconds=time.perf_counter()-t0
    scored=valid.join(score_pairs(model,valid),on=["s1_id","target_id"])
    y=scored["is_match"].cast(pl.Int8).to_numpy()
    p=scored["match_probability"].to_numpy()
    precision,recall,thresholds=precision_recall_curve(y,p)

    operating=[]
    for threshold in (0.5,0.7,0.8,0.9,0.95,0.98,0.99):
        pred=p>=threshold
        tp=int(((pred==1)&(y==1)).sum())
        fp=int(((pred==1)&(y==0)).sum())
        fn=int(((pred==0)&(y==1)).sum())
        operating.append({
            "threshold":threshold,"tp":tp,"fp":fp,"fn":fn,
            "precision":tp/(tp+fp) if tp+fp else 1.0,
            "recall":tp/(tp+fn) if tp+fn else 0.0,
        })

    result={
        "purpose":"Stage 7 bounded LightGBM pair-classifier baseline",
        "country":country,
        "train_pairs":train.height,
        "validation_pairs":valid.height,
        "train_positives":int(train["is_match"].sum()),
        "validation_positives":int(valid["is_match"].sum()),
        "best_iteration":getattr(model,"best_iteration_",None),
        "train_seconds":train_seconds,
        "validation_roc_auc":float(roc_auc_score(y,p)),
        "validation_average_precision":float(average_precision_score(y,p)),
        "probability_min":float(p.min()),
        "probability_max":float(p.max()),
        "operating_points":operating,
    }
    out=Path("artifacts")/"stage7_lightgbm_baseline.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    print(f"Saved: {out}")
    return result


def main()->None:
    run()


if __name__=="__main__":
    main()

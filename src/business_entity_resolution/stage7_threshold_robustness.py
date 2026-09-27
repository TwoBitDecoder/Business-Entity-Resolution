"""Stage 7 threshold robustness across deterministic S1-grouped splits."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .features import build_pair_features
from .hybrid_candidates import combine_fuzzy_candidates
from .metrics import entity_macro_fbeta
from .model import score_pairs, train_pair_classifier
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .stage7_entity_threshold_sweep import THRESHOLDS
from .training_data import add_grouped_split, label_candidate_features

SEEDS=(11,23,42,67,101)


def run(root: str | Path="artifacts/preprocessed/train", *, country: str="India",
        query_count: int=1000, target_limit_per_source: int=250000) -> dict:
    base=Path(root)
    queries=_load(_partition_path(base,"source1",country),query_count)
    targets=pl.concat([
        _load(_partition_path(base,"source2",country),target_limit_per_source),
        _load(_partition_path(base,"source3",country),target_limit_per_source),
    ])
    print(f"{country}: retrieving frozen candidates once...",flush=True)
    name=sparse_matmul_topk(queries,targets,config=RetrievalConfig(top_k=20),text_column="name_compact")
    address=sparse_matmul_topk(queries,targets,config=RetrievalConfig(top_k=20),text_column="address_norm")
    candidates=combine_fuzzy_candidates(name,address,final_top_k=40)
    features=build_pair_features(candidates,queries,targets)
    truth=load_truth("data/train/train_ground_truth.tsv",queries["entity_id"].to_list())
    labelled=label_candidate_features(features,truth)

    runs=[]
    threshold_scores={t:[] for t in THRESHOLDS}
    for seed in SEEDS:
        data=add_grouped_split(labelled,seed=seed)
        train=data.filter(pl.col("split")=="train")
        valid=data.filter(pl.col("split")=="validation")
        model=train_pair_classifier(train,valid,seed=seed)
        scored=score_pairs(model,valid)
        ids=valid["s1_id"].unique().to_list()
        valid_truth=truth.filter(pl.col("s1_id").is_in(ids))
        sweep=[]
        for t in THRESHOLDS:
            score=entity_macro_fbeta(ids,valid_truth,scored,threshold=t,beta=.5)
            threshold_scores[t].append(score)
            sweep.append((t,score))
        best=max(sweep,key=lambda x:(x[1],x[0]))
        runs.append({"seed":seed,"validation_entities":len(ids),
                     "best_threshold":best[0],"best_macro_f0_5":best[1]})
        print(f"seed {seed}: threshold={best[0]:.2f}, macro F0.5={best[1]:.6f}",flush=True)

    aggregate=[
        {"threshold":t,
         "mean_macro_f0_5":sum(scores)/len(scores),
         "min_macro_f0_5":min(scores),
         "max_macro_f0_5":max(scores)}
        for t,scores in threshold_scores.items()
    ]
    best_mean=max(aggregate,key=lambda x:(x["mean_macro_f0_5"],x["threshold"]))
    result={"purpose":"Stage 7 threshold robustness across grouped splits",
            "country":country,"seeds":list(SEEDS),"runs":runs,
            "best_mean_threshold":best_mean["threshold"],
            "best_mean_macro_f0_5":best_mean["mean_macro_f0_5"],
            "aggregate":aggregate}
    out=Path("artifacts")/"stage7_threshold_robustness.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    print(f"Saved: {out}")
    return result


def main()->None:
    run()


if __name__=="__main__":
    main()

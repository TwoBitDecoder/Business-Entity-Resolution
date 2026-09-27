"""Stage 7 exact entity-level macro F0.5 threshold sweep."""
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
from .training_data import add_grouped_split, label_candidate_features

THRESHOLDS = [round(x / 100, 2) for x in range(1, 100)]


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
    print(f"{country}: retrieving frozen candidates...",flush=True)
    name=sparse_matmul_topk(queries,targets,config=RetrievalConfig(top_k=20),text_column="name_compact")
    address=sparse_matmul_topk(queries,targets,config=RetrievalConfig(top_k=20),text_column="address_norm")
    candidates=combine_fuzzy_candidates(name,address,final_top_k=40)

    features=build_pair_features(candidates,queries,targets)
    truth=load_truth("data/train/train_ground_truth.tsv",queries["entity_id"].to_list())
    labelled=add_grouped_split(label_candidate_features(features,truth))
    train=labelled.filter(pl.col("split")=="train")
    valid=labelled.filter(pl.col("split")=="validation")
    model=train_pair_classifier(train,valid)
    scored=score_pairs(model,valid)
    validation_ids=valid["s1_id"].unique().to_list()
    validation_truth=truth.filter(pl.col("s1_id").is_in(validation_ids))

    print(f"{country}: sweeping exact entity macro F0.5...",flush=True)
    sweep=[
        {
            "threshold":t,
            "macro_f0_5":entity_macro_fbeta(
                validation_ids,validation_truth,scored,threshold=t,beta=.5
            ),
        }
        for t in THRESHOLDS
    ]
    best=max(sweep,key=lambda x:(x["macro_f0_5"],x["threshold"]))
    result={
        "purpose":"Stage 7 exact entity-level macro F0.5 threshold sweep",
        "country":country,
        "validation_entities":len(validation_ids),
        "validation_truth_pairs":validation_truth.height,
        "validation_candidate_pairs":valid.height,
        "best_threshold":best["threshold"],
        "best_macro_f0_5":best["macro_f0_5"],
        "sweep":sweep,
    }
    out=Path("artifacts")/"stage7_entity_threshold_sweep.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    print(f"Saved: {out}")
    return result


def main()->None:
    run()


if __name__=="__main__":
    main()

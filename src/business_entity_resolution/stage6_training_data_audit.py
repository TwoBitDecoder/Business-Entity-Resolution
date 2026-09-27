"""Stage 6 bounded real-data training-dataset audit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .features import build_pair_features
from .hybrid_candidates import combine_fuzzy_candidates
from .r5_hybrid_benchmark import _load
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk
from .training_data import add_grouped_split, label_candidate_features, validate_grouped_split


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1000,
    target_limit_per_source: int = 250000,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    base = Path(root)
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(base, "source2", country), target_limit_per_source),
        _load(_partition_path(base, "source3", country), target_limit_per_source),
    ])

    print(f"{country}: retrieving candidates...", flush=True)
    t0 = time.perf_counter()
    name = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=20), text_column="name_compact"
    )
    address = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=20), text_column="address_norm"
    )
    candidates = combine_fuzzy_candidates(name, address, final_top_k=40)
    retrieval_seconds = time.perf_counter() - t0

    print(f"{country}: building and labelling features...", flush=True)
    t0 = time.perf_counter()
    features = build_pair_features(candidates, queries, targets)
    truth = load_truth("data/train/train_ground_truth.tsv", queries["entity_id"].to_list())
    labelled = label_candidate_features(features, truth)
    split = add_grouped_split(
        labelled, validation_fraction=validation_fraction, seed=seed
    )
    validate_grouped_split(split)
    build_seconds = time.perf_counter() - t0

    entity_split = split.select("s1_id", "split").unique()
    leakage = (
        entity_split.group_by("s1_id")
        .agg(pl.col("split").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )

    stats = {}
    for split_name in ("train", "validation"):
        part = split.filter(pl.col("split") == split_name)
        positives = int(part["is_match"].sum())
        stats[split_name] = {
            "entities": part["s1_id"].n_unique(),
            "candidate_pairs": part.height,
            "positive_pairs": positives,
            "negative_pairs": part.height - positives,
            "positive_rate": positives / part.height if part.height else 0.0,
        }

    positives_per_entity = (
        split.filter(pl.col("is_match"))
        .group_by("s1_id").len()
    )
    result = {
        "purpose": "Stage 6 bounded labelled training-data audit",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "candidate_rows": candidates.height,
        "validation_fraction_requested": validation_fraction,
        "seed": seed,
        "s1_split_leakage_count": leakage,
        "splits": stats,
        "positive_entities_in_candidates": positives_per_entity.height,
        "max_positive_candidates_per_entity": (
            int(positives_per_entity["len"].max()) if positives_per_entity.height else 0
        ),
        "retrieval_seconds": retrieval_seconds,
        "feature_label_split_seconds": build_seconds,
    }
    out = Path("artifacts") / "stage6_training_data_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()

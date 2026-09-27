"""R7: conservative minimum-similarity pruning sweep.

Builds each validated char-TFIDF target index once, then reuses it across
thresholds. This keeps the representation and Top-K fixed while avoiding ten
expensive target refits.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .r5_hybrid_benchmark import _load, _pair_recall
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .reusable_sparse_index import SparseTopKIndex

THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20)


def run(
    root: str | Path = "artifacts/preprocessed/train",
    *,
    country: str = "India",
    query_count: int = 1000,
    target_limit_per_source: int = 250000,
) -> dict:
    base = Path(root)
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat(
        [
            _load(_partition_path(base, "source2", country), target_limit_per_source),
            _load(_partition_path(base, "source3", country), target_limit_per_source),
        ]
    )
    truth = load_truth("data/train/train_ground_truth.tsv", queries["entity_id"].to_list())
    truth = truth.join(
        targets.select(pl.col("entity_id").alias("target_id")).unique(),
        on="target_id",
        how="inner",
    )

    print(f"{country}: building reusable name index...", flush=True)
    t0 = time.perf_counter()
    name_index = SparseTopKIndex(
        targets,
        text_column="name_compact",
        config=RetrievalConfig(top_k=20),
    )
    name_build_seconds = time.perf_counter() - t0
    name_results = {}
    for threshold in THRESHOLDS:
        t0 = time.perf_counter()
        name_results[threshold] = (
            name_index.query(queries, min_similarity=threshold),
            time.perf_counter() - t0,
        )
        print(f"{country}: name threshold {threshold:.2f} done", flush=True)
    del name_index
    gc.collect()

    print(f"{country}: building reusable address index...", flush=True)
    t0 = time.perf_counter()
    address_index = SparseTopKIndex(
        targets,
        text_column="address_norm",
        config=RetrievalConfig(top_k=20),
    )
    address_build_seconds = time.perf_counter() - t0
    address_results = {}
    for threshold in THRESHOLDS:
        t0 = time.perf_counter()
        address_results[threshold] = (
            address_index.query(queries, min_similarity=threshold),
            time.perf_counter() - t0,
        )
        print(f"{country}: address threshold {threshold:.2f} done", flush=True)
    del address_index
    gc.collect()

    rows = []
    for threshold in THRESHOLDS:
        name, name_seconds = name_results[threshold]
        address, address_seconds = address_results[threshold]
        hybrid = combine_fuzzy_candidates(name, address, final_top_k=40)
        recall = _pair_recall(hybrid, truth)
        rows.append(
            {
                "min_similarity": threshold,
                "name_candidate_rows": name.height,
                "address_candidate_rows": address.height,
                "hybrid_candidate_rows": hybrid.height,
                "truth_pairs_retrieved": round((recall or 0) * truth.height),
                "pair_recall": recall,
                "name_query_seconds": name_seconds,
                "address_query_seconds": address_seconds,
                "query_seconds": name_seconds + address_seconds,
            }
        )

    return {
        "purpose": "R7 minimum-similarity pruning sweep with reusable indexes",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "reachable_truth_pairs": truth.height,
        "name_top_k": 20,
        "address_top_k": 20,
        "final_top_k": 40,
        "name_index_build_seconds": name_build_seconds,
        "address_index_build_seconds": address_build_seconds,
        "results": rows,
    }


def main() -> None:
    result = run()
    out = Path("artifacts") / "r7_similarity_threshold_sweep.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

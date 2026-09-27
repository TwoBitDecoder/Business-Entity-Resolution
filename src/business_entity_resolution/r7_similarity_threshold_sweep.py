"""R7: conservative minimum-similarity pruning sweep.

Keeps the validated char TF-IDF representation and Top-K fixed while measuring
whether sparse-dot-topn thresholds can reduce runtime without materially
reducing reachable-truth recall.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .r5_hybrid_benchmark import _load, _pair_recall
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk


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

    rows = []
    for threshold in THRESHOLDS:
        cfg = RetrievalConfig(top_k=20, min_similarity=threshold)

        t0 = time.perf_counter()
        name = sparse_matmul_topk(
            queries, targets, config=cfg, text_column="name_compact"
        )
        name_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        address = sparse_matmul_topk(
            queries, targets, config=cfg, text_column="address_norm"
        )
        address_seconds = time.perf_counter() - t0

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
                "name_seconds": name_seconds,
                "address_seconds": address_seconds,
                "retrieval_seconds": name_seconds + address_seconds,
            }
        )

    return {
        "purpose": "R7 minimum-similarity pruning sweep",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "reachable_truth_pairs": truth.height,
        "name_top_k": 20,
        "address_top_k": 20,
        "final_top_k": 40,
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

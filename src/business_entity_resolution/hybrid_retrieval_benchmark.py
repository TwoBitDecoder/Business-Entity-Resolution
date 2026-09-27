"""Stage 4C.1 benchmark: compare name-only and hybrid retrieval recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk
from .hybrid_retrieval import retrieve_hybrid_topk
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth, recall_at_k


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def run_hybrid_benchmark(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *,
    country: str = "India",
    query_limit: int = 1_000,
    target_limit_per_source: int = 250_000,
    name_k: int = 20,
    address_k: int = 10,
    final_k: int = 30,
    n_jobs: int = -1,
) -> dict:
    base = Path(root)
    q = _load(_partition_path(base, "source1", country), query_limit)
    s2 = _load(_partition_path(base, "source2", country), target_limit_per_source)
    s3 = _load(_partition_path(base, "source3", country), target_limit_per_source)
    targets = pl.concat([s2, s3], how="vertical")

    truth_all = load_truth(ground_truth_path, q["entity_id"].to_list())
    target_ids = targets.select(pl.col("entity_id").alias("target_id")).unique()
    truth = truth_all.join(target_ids, on="target_id", how="inner")

    name = retrieve_topk(
        q.select("entity_id", "name_compact"),
        targets.select("entity_id", "name_compact"),
        config=RetrievalConfig(top_k=name_k, n_jobs=n_jobs),
    )
    hybrid = retrieve_hybrid_topk(
        q, targets, name_k=name_k, address_k=address_k, final_k=final_k, n_jobs=n_jobs
    )
    hybrid_for_metric = hybrid.rename({"similarity": "name_similarity"})

    return {
        "country": country,
        "query_rows": q.height,
        "target_rows": targets.height,
        "truth_pairs_in_target_pool": truth.height,
        "name_only": {
            "candidate_rows": name.height,
            **recall_at_k(name, truth, ks=(5, 10, 20)),
        },
        "hybrid": {
            "name_k": name_k,
            "address_k": address_k,
            "final_k": final_k,
            "candidate_rows": hybrid.height,
            **recall_at_k(hybrid_for_metric, truth, ks=(5, 10, 20, final_k)),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1_000)
    p.add_argument("--targets-per-source", type=int, default=250_000)
    p.add_argument("--name-k", type=int, default=20)
    p.add_argument("--address-k", type=int, default=10)
    p.add_argument("--final-k", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    result = run_hybrid_benchmark(
        country=a.country, query_limit=a.queries,
        target_limit_per_source=a.targets_per_source,
        name_k=a.name_k, address_k=a.address_k, final_k=a.final_k, n_jobs=a.n_jobs,
    )
    out = Path("artifacts/stage4c1_hybrid.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

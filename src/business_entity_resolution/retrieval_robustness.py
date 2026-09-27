"""Stage 4C.2: reproducible robustness benchmark across countries and samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk
from .hybrid_retrieval import retrieve_hybrid_topk
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth, recall_at_k


def _sample(path: Path, n: int, seed: int) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .collect(engine="streaming")
    )
    return frame.sample(n=min(n, frame.height), seed=seed, shuffle=True)


def _targets(path: Path, n: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .head(n)
        .collect(engine="streaming")
    )


def evaluate_sample(
    root: Path,
    ground_truth_path: Path,
    *,
    country: str,
    seed: int,
    query_count: int,
    target_limit_per_source: int,
    name_k: int,
    address_k: int,
    final_k: int,
    n_jobs: int,
) -> dict:
    q = _sample(_partition_path(root, "source1", country), query_count, seed)
    targets = pl.concat([
        _targets(_partition_path(root, "source2", country), target_limit_per_source),
        _targets(_partition_path(root, "source3", country), target_limit_per_source),
    ], how="vertical")

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
    hybrid_metric = hybrid.rename({"similarity": "name_similarity"})
    missed = (
        truth.join(
            hybrid.select("s1_id", "target_id").unique(),
            on=["s1_id", "target_id"],
            how="anti",
        ).height
    )
    return {
        "country": country,
        "seed": seed,
        "query_rows": q.height,
        "target_rows": targets.height,
        "truth_pairs_for_queries": truth_all.height,
        "truth_pairs_in_target_pool": truth.height,
        "target_pool_truth_coverage": truth.height / truth_all.height if truth_all.height else None,
        "name_only": {
            "candidate_rows": name.height,
            **recall_at_k(name, truth, ks=(5, 10, 20)),
        },
        "hybrid": {
            "candidate_rows": hybrid.height,
            "missed_truth_pairs_at_final_k": missed,
            **recall_at_k(hybrid_metric, truth, ks=(5, 10, 20, final_k)),
        },
    }


def run_robustness(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *,
    countries: tuple[str, ...] = ("India", "US"),
    seeds: tuple[int, ...] = (17, 43),
    query_count: int = 500,
    target_limit_per_source: int = 250_000,
    name_k: int = 20,
    address_k: int = 10,
    final_k: int = 30,
    n_jobs: int = -1,
) -> dict:
    base, gt = Path(root), Path(ground_truth_path)
    runs = [
        evaluate_sample(
            base, gt, country=country, seed=seed, query_count=query_count,
            target_limit_per_source=target_limit_per_source, name_k=name_k,
            address_k=address_k, final_k=final_k, n_jobs=n_jobs,
        )
        for country in countries
        for seed in seeds
    ]
    return {
        "config": {
            "countries": list(countries), "seeds": list(seeds),
            "queries_per_run": query_count,
            "targets_per_source": target_limit_per_source,
            "name_k": name_k, "address_k": address_k, "final_k": final_k,
        },
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--targets-per-source", type=int, default=250_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 43])
    p.add_argument("--countries", nargs="+", default=["India", "US"])
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    result = run_robustness(
        countries=tuple(a.countries), seeds=tuple(a.seeds),
        query_count=a.queries, target_limit_per_source=a.targets_per_source,
        n_jobs=a.n_jobs,
    )
    out = Path("artifacts/stage4c2_robustness.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

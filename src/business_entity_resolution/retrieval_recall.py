"""Stage 4B.3: ground-truth Recall@K evaluation for bounded retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk, validate_bound
from .retrieval_benchmark_runner import _partition_path, load_head


def load_truth(path: str | Path, query_ids: list[str]) -> pl.DataFrame:
    """Expand ground truth only for the benchmark query IDs."""
    if not query_ids:
        return pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String})
    return (
        pl.scan_csv(path, separator="\t", infer_schema=False)
        .select(
            pl.col("source1_entity_id").cast(pl.String).alias("s1_id"),
            pl.col("matched_entity_ids").cast(pl.String).fill_null(""),
        )
        .filter(pl.col("s1_id").is_in(query_ids))
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("target_id"))
        .explode("target_id", empty_as_null=True)
        .filter(pl.col("target_id").is_not_null() & (pl.col("target_id") != ""))
        .select("s1_id", pl.col("target_id").str.strip_chars())
        .collect(engine="streaming")
    )


def recall_at_k(candidates: pl.DataFrame, truth: pl.DataFrame, ks=(5, 10, 20)) -> dict:
    """Positive-pair recall among truth IDs present in the benchmark target pool."""
    if truth.is_empty():
        return {f"recall_at_{k}": None for k in ks}

    ranked = candidates.with_columns(
        pl.col("name_similarity")
        .rank(method="ordinal", descending=True)
        .over("s1_id")
        .alias("rank")
    )
    total = truth.height
    result = {}
    for k in ks:
        retrieved = (
            ranked.filter(pl.col("rank") <= k)
            .select("s1_id", "target_id")
            .join(truth, on=["s1_id", "target_id"], how="inner")
            .unique()
            .height
        )
        result[f"recall_at_{k}"] = retrieved / total
    return result


def run_recall_benchmark(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *,
    country: str = "India",
    query_limit: int = 1_000,
    target_limit_per_source: int = 250_000,
    top_k: int = 20,
    n_jobs: int = -1,
) -> dict:
    base = Path(root)
    queries = load_head(_partition_path(base, "source1", country), query_limit)
    s2 = load_head(_partition_path(base, "source2", country), target_limit_per_source)
    s3 = load_head(_partition_path(base, "source3", country), target_limit_per_source)
    targets = pl.concat([s2, s3], how="vertical")

    cfg = RetrievalConfig(top_k=top_k, n_jobs=n_jobs)
    candidates = retrieve_topk(queries, targets, config=cfg)
    nonempty = queries.filter(pl.col("name_compact") != "").height
    validate_bound(candidates, nonempty, top_k)

    query_ids = queries["entity_id"].to_list()
    truth_all = load_truth(ground_truth_path, query_ids)
    target_ids = targets.select(pl.col("entity_id").alias("target_id")).unique()
    truth_in_pool = truth_all.join(target_ids, on="target_id", how="inner")

    metrics = recall_at_k(candidates, truth_in_pool, ks=tuple(k for k in (5, 10, 20) if k <= top_k))
    truth_entities = truth_in_pool["s1_id"].n_unique() if truth_in_pool.height else 0
    return {
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "top_k": top_k,
        "candidate_rows": candidates.height,
        "truth_pairs_for_queries": truth_all.height,
        "truth_pairs_in_target_pool": truth_in_pool.height,
        "truth_entities_in_target_pool": truth_entities,
        "target_pool_truth_coverage": (
            truth_in_pool.height / truth_all.height if truth_all.height else None
        ),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--targets-per-source", type=int, default=250_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    result = run_recall_benchmark(
        country=args.country,
        query_limit=args.queries,
        target_limit_per_source=args.targets_per_source,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
    )
    out = Path("artifacts/stage4b3_recall.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

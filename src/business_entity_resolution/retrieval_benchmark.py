"""Stage 2: memory-conscious normalization and candidate-recall benchmark.

This intentionally benchmarks cheap exact blocking rules before fuzzy retrieval.
It uses Polars lazy scans/joins so multi-million-row TSVs are not materialized
as Python objects.

Run:
    python -m business_entity_resolution.retrieval_benchmark
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata

import polars as pl

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRAIN_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
    "ground_truth": "train_ground_truth.tsv",
}


def _scan(path: Path, columns: list[str]) -> pl.LazyFrame:
    lf = pl.scan_csv(
        path, separator="\t", infer_schema=False, null_values=[],
        low_memory=True, has_header=True,
    )
    missing = [c for c in columns if c not in lf.collect_schema().names()]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    return lf.select(pl.col(c).cast(pl.String).fill_null("").alias(c) for c in columns)


def _norm_expr(column: str) -> pl.Expr:
    # Unicode-aware, language-agnostic normalization. Polars handles the
    # expensive vectorized string work; punctuation/space removal gives a
    # compact key robust to formatting variation.
    return (
        pl.col(column)
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _compact_expr(column: str) -> pl.Expr:
    return _norm_expr(column).str.replace_all(" ", "")


def normalized(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select(
        "entity_id",
        pl.col("country").str.strip_chars().alias("country"),
        _norm_expr("business_name").alias("name_norm"),
        _compact_expr("business_name").alias("name_compact"),
        _norm_expr("business_address").alias("address_norm"),
    )


def positives(gt: pl.LazyFrame) -> pl.LazyFrame:
    return (
        gt.with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("target_id"))
        .explode("target_id", empty_as_null=True)
        .with_columns(pl.col("target_id").str.strip_chars())
        .filter(pl.col("target_id") != "")
        .select(pl.col("source1_entity_id").alias("s1_id"), "target_id")
    )


def _candidate_rule(
    left: pl.LazyFrame, right: pl.LazyFrame, key: str, rule: str
) -> pl.LazyFrame:
    l = left.filter(pl.col(key) != "").select(
        pl.col("entity_id").alias("s1_id"), "country", pl.col(key).alias("key")
    )
    r = right.filter(pl.col(key) != "").select(
        pl.col("entity_id").alias("target_id"), "country", pl.col(key).alias("key")
    )
    return (
        l.join(r, on=["country", "key"], how="inner")
        .select("s1_id", "target_id")
        .with_columns(pl.lit(rule).alias("rule"))
    )


def _metric(rule_pairs: pl.LazyFrame, truth: pl.LazyFrame, total_truth: int) -> dict:
    unique = rule_pairs.select("s1_id", "target_id").unique()
    counts = unique.select(
        pl.len().alias("candidates"),
        pl.col("s1_id").n_unique().alias("covered_s1"),
    ).collect(engine="streaming").row(0, named=True)
    hits = (
        truth.join(unique, on=["s1_id", "target_id"], how="semi")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    return {
        "candidate_pairs": int(counts["candidates"]),
        "covered_s1": int(counts["covered_s1"]),
        "true_pairs_retrieved": int(hits),
        "positive_pair_recall": (float(hits) / total_truth) if total_truth else 0.0,
    }


def _block_stats(frame: pl.LazyFrame, key: str) -> dict:
    sizes = (
        frame.filter(pl.col(key) != "")
        .group_by(["country", key])
        .len()
        .select("len")
        .collect(engine="streaming")
    )
    if sizes.is_empty():
        return {}
    q = sizes.select(
        pl.col("len").median().alias("median"),
        pl.col("len").quantile(0.90, interpolation="nearest").alias("p90"),
        pl.col("len").quantile(0.95, interpolation="nearest").alias("p95"),
        pl.col("len").quantile(0.99, interpolation="nearest").alias("p99"),
        pl.col("len").max().alias("max"),
    ).row(0, named=True)
    return {k: int(v) for k, v in q.items()}


def build_benchmark(data_dir: str | Path = "data") -> dict:
    root = Path(data_dir) / "train"
    paths = {k: root / v for k, v in TRAIN_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    s1 = normalized(_scan(paths["source1"], SOURCE_COLUMNS))
    s2 = normalized(_scan(paths["source2"], SOURCE_COLUMNS))
    s3 = normalized(_scan(paths["source3"], SOURCE_COLUMNS))
    targets = pl.concat([s2, s3])

    gt = _scan(paths["ground_truth"], ["source1_entity_id", "matched_entity_ids"])
    truth = positives(gt)
    total_truth = int(truth.select(pl.len()).collect(engine="streaming").item())

    rules = {
        "exact_name": _candidate_rule(s1, targets, "name_norm", "exact_name"),
        "compact_name": _candidate_rule(s1, targets, "name_compact", "compact_name"),
        "exact_address": _candidate_rule(s1, targets, "address_norm", "exact_address"),
    }
    metrics = {name: _metric(pairs, truth, total_truth) for name, pairs in rules.items()}
    union = pl.concat(list(rules.values())).select("s1_id", "target_id").unique()
    metrics["union"] = _metric(union, truth, total_truth)

    return {
        "total_positive_pairs": total_truth,
        "rules": metrics,
        "target_block_sizes": {
            "name_norm": _block_stats(targets, "name_norm"),
            "name_compact": _block_stats(targets, "name_compact"),
            "address_norm": _block_stats(targets, "address_norm"),
        },
        "notes": [
            "All blocking is country-scoped.",
            "This stage measures exact/compact retrieval only; fuzzy retrieval is intentionally deferred.",
            "Candidate pair counts can be large for common names/addresses; inspect block-size tails before materializing candidates.",
        ],
    }


def main() -> None:
    result = build_benchmark()
    output = Path("artifacts") / "retrieval_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=" * 68)
    print("STAGE 2 — CANDIDATE RETRIEVAL BENCHMARK")
    print("=" * 68)
    print(f"Ground-truth positive pairs: {result['total_positive_pairs']:,}")
    for name, m in result["rules"].items():
        print(
            f"{name:>14}: recall={m['positive_pair_recall']:.4%}  "
            f"pairs={m['candidate_pairs']:,}  covered_S1={m['covered_s1']:,}"
        )
    print("\nTarget block sizes:")
    for key, stats in result["target_block_sizes"].items():
        print(f"  {key}: {stats}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

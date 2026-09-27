"""Stage 3 residual candidate profiler.

Profiles the positive pairs missed by exact blocking and measures inexpensive
fuzzy-friendly signals without constructing an all-pairs fuzzy matrix.

Run:
    python -m business_entity_resolution.residual_profiler
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .retrieval_benchmark import (
    SOURCE_COLUMNS, TRAIN_FILES, _candidate_rule, _scan, normalized, positives
)


def _prefix_expr(col: str, n: int) -> pl.Expr:
    return pl.col(col).str.slice(0, n)


def _numeric_expr(col: str) -> pl.Expr:
    return pl.col(col).str.extract_all(r"\d+").list.join("|")


def enriched(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        _prefix_expr("name_compact", 4).alias("name_prefix4"),
        _prefix_expr("name_compact", 6).alias("name_prefix6"),
        _numeric_expr("address_norm").alias("address_numbers"),
        pl.col("name_compact").str.len_chars().alias("name_len"),
    )


def _rule(left: pl.LazyFrame, right: pl.LazyFrame, key: str) -> pl.LazyFrame:
    return _candidate_rule(left, right, key, key).select("s1_id", "target_id")


def _stats(pairs: pl.LazyFrame, truth: pl.LazyFrame, residual: pl.LazyFrame, total: int, residual_n: int) -> dict:
    pairs = pairs.unique()
    total_hits = truth.join(pairs, on=["s1_id", "target_id"], how="semi").select(pl.len()).collect(engine="streaming").item()
    residual_hits = residual.join(pairs, on=["s1_id", "target_id"], how="semi").select(pl.len()).collect(engine="streaming").item()
    candidate_n = pairs.select(pl.len()).collect(engine="streaming").item()
    return {
        "candidate_pairs": int(candidate_n),
        "total_true_pairs_retrieved": int(total_hits),
        "total_positive_recall": float(total_hits / total) if total else 0.0,
        "residual_true_pairs_retrieved": int(residual_hits),
        "residual_recall": float(residual_hits / residual_n) if residual_n else 0.0,
    }


def build_profile(data_dir: str | Path = "data") -> dict:
    root = Path(data_dir) / "train"
    paths = {k: root / v for k, v in TRAIN_FILES.items()}
    s1 = enriched(normalized(_scan(paths["source1"], SOURCE_COLUMNS)))
    s2 = enriched(normalized(_scan(paths["source2"], SOURCE_COLUMNS)))
    s3 = enriched(normalized(_scan(paths["source3"], SOURCE_COLUMNS)))
    targets = pl.concat([s2, s3])

    gt = _scan(paths["ground_truth"], ["source1_entity_id", "matched_entity_ids"])
    truth = positives(gt)
    total = int(truth.select(pl.len()).collect(engine="streaming").item())

    exact = pl.concat([
        _rule(s1, targets, "name_norm"),
        _rule(s1, targets, "name_compact"),
        _rule(s1, targets, "address_norm"),
    ]).unique()
    residual = truth.join(exact, on=["s1_id", "target_id"], how="anti")
    residual_n = int(residual.select(pl.len()).collect(engine="streaming").item())

    # Prefix keys are deliberately country-scoped through _candidate_rule.
    # Prefix-4 may be large, so it is diagnostic; we do not persist candidates.
    rules = {
        "name_prefix6": _rule(s1, targets, "name_prefix6"),
        "address_numbers": _rule(
            s1.filter(pl.col("address_numbers") != ""),
            targets.filter(pl.col("address_numbers") != ""),
            "address_numbers",
        ),
    }
    metrics = {name: _stats(p, truth, residual, total, residual_n) for name, p in rules.items()}

    return {
        "total_positive_pairs": total,
        "exact_union_retrieved": total - residual_n,
        "residual_positive_pairs": residual_n,
        "rules": metrics,
        "warning": (
            "These are diagnostics, not final blockers. Prefix-4 is disabled at full scale because "
            "its many-to-many join can exhaust memory. Stage 4 will use capped sparse top-K retrieval."
        ),
    }


def main() -> None:
    result = build_profile()
    out = Path("artifacts") / "residual_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("=" * 68)
    print("STAGE 3 — RESIDUAL RETRIEVAL PROFILER")
    print("=" * 68)
    print(f"Positive pairs: {result['total_positive_pairs']:,}")
    print(f"Missed by exact union: {result['residual_positive_pairs']:,}")
    for name, m in result["rules"].items():
        print(
            f"{name:>15}: residual_recall={m['residual_recall']:.3%}  "
            f"total_recall={m['total_positive_recall']:.3%}  "
            f"pairs={m['candidate_pairs']:,}"
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

"""Memory-conscious local dataset audit for multi-million-row TSV files.

The audit uses Polars lazy scans for source-level statistics and only collects the
columns/rows needed for ground-truth validation. It prints aggregate data only.

Run:
    python -m business_entity_resolution.eda
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

TRAIN_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
    "ground_truth": "train_ground_truth.tsv",
}
TEST_FILES = {
    "source1": "test_source1.tsv",
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _require_files(root: Path, names: dict[str, str]) -> dict[str, Path]:
    paths = {key: root / name for key, name in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))
    return paths


def _scan_source(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(
        path,
        separator="\t",
        has_header=True,
        infer_schema=False,
        null_values=[],
        low_memory=True,
    )
    schema = lf.collect_schema()
    missing = [c for c in SOURCE_COLUMNS if c not in schema.names()]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    return lf.select(
        pl.col(c).cast(pl.String).fill_null("").alias(c) for c in SOURCE_COLUMNS
    )


def _scan_gt(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(
        path,
        separator="\t",
        has_header=True,
        infer_schema=False,
        null_values=[],
        low_memory=True,
    )
    required = ["source1_entity_id", "matched_entity_ids"]
    schema = lf.collect_schema()
    missing = [c for c in required if c not in schema.names()]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    return lf.select(pl.col(c).cast(pl.String).fill_null("").alias(c) for c in required)


def _pct(n: int, d: int) -> str:
    return "0.00%" if d == 0 else f"{100 * n / d:.2f}%"


def _source_stats(label: str, lf: pl.LazyFrame) -> list[str]:
    summary = lf.select(
        pl.len().alias("n"),
        (pl.len() - pl.col("entity_id").n_unique()).alias("duplicate_ids"),
        pl.col("business_name").str.strip_chars().eq("").sum().alias("missing_name"),
        pl.col("business_address").str.strip_chars().eq("").sum().alias("missing_address"),
        pl.col("country").str.strip_chars().eq("").sum().alias("missing_country"),
    ).collect(engine="streaming").row(0, named=True)
    countries = (
        lf.select(
            pl.when(pl.col("country").str.strip_chars().eq(""))
            .then(pl.lit("<missing>"))
            .otherwise(pl.col("country").str.strip_chars())
            .alias("country")
        )
        .group_by("country")
        .len()
        .sort("country")
        .collect(engine="streaming")
    )
    n = int(summary["n"])
    country_text = ", ".join(
        f"{row['country']}={row['len']:,}" for row in countries.iter_rows(named=True)
    )
    return [
        f"{label}: {n:,} records",
        f"  duplicate entity_id: {int(summary['duplicate_ids']):,}",
        f"  missing business_name: {int(summary['missing_name']):,} ({_pct(int(summary['missing_name']), n)})",
        f"  missing business_address: {int(summary['missing_address']):,} ({_pct(int(summary['missing_address']), n)})",
        f"  missing country: {int(summary['missing_country']):,} ({_pct(int(summary['missing_country']), n)})",
        f"  countries: {country_text}",
    ]


def _ground_truth_stats(
    s1: pl.LazyFrame, s2: pl.LazyFrame, s3: pl.LazyFrame, gt: pl.LazyFrame
) -> list[str]:
    # Explode ground truth lazily instead of constructing Python sets containing
    # millions of string IDs.
    gt_clean = gt.with_columns(
        pl.col("source1_entity_id").str.strip_chars(),
        pl.col("matched_entity_ids").str.strip_chars(),
    )
    gt_summary = gt_clean.select(
        pl.len().alias("n"),
        (pl.len() - pl.col("source1_entity_id").n_unique()).alias("duplicate_gt_s1"),
        pl.col("matched_entity_ids").eq("").sum().alias("singletons"),
    ).collect(engine="streaming").row(0, named=True)

    matches = (
        gt_clean.filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("target_id"))
        .explode("target_id")
        .with_columns(pl.col("target_id").str.strip_chars())
        .filter(pl.col("target_id") != "")
        .select("source1_entity_id", "target_id")
    )
    mult = (
        matches.group_by("source1_entity_id")
        .len()
        .select(
            pl.col("len").eq(1).sum().alias("one"),
            pl.col("len").eq(2).sum().alias("two"),
            pl.col("len").ge(3).sum().alias("three_plus"),
            pl.col("len").sum().alias("positive_pairs"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    s1_ids = s1.select(pl.col("entity_id").alias("source1_entity_id")).unique()
    candidate_ids = pl.concat(
        [s2.select(pl.col("entity_id").alias("target_id")),
         s3.select(pl.col("entity_id").alias("target_id"))]
    ).unique()

    missing_gt_s1 = (
        s1_ids.join(gt_clean.select("source1_entity_id").unique(),
                    on="source1_entity_id", how="anti")
        .select(pl.len()).collect(engine="streaming").item()
    )
    unknown_s1 = (
        gt_clean.select("source1_entity_id").join(s1_ids, on="source1_entity_id", how="anti")
        .select(pl.len()).collect(engine="streaming").item()
    )
    unknown_targets = (
        matches.select("target_id").join(candidate_ids, on="target_id", how="anti")
        .select(pl.len()).collect(engine="streaming").item()
    )

    n = int(gt_summary["n"])
    singletons = int(gt_summary["singletons"])
    one, two, three = int(mult["one"] or 0), int(mult["two"] or 0), int(mult["three_plus"] or 0)
    positive = int(mult["positive_pairs"] or 0)
    return [
        f"Ground-truth rows: {n:,}",
        f"  S1 IDs missing from ground truth: {int(missing_gt_s1):,}",
        f"  duplicate S1 rows in ground truth: {int(gt_summary['duplicate_gt_s1']):,}",
        f"  unknown S1 IDs in ground truth: {int(unknown_s1):,}",
        f"  unknown matched S2/S3 IDs: {int(unknown_targets):,}",
        f"  singleton S1 entities: {singletons:,} ({_pct(singletons, n)})",
        f"  exactly 1 match: {one:,} ({_pct(one, n)})",
        f"  exactly 2 matches: {two:,} ({_pct(two, n)})",
        f"  3+ matches: {three:,} ({_pct(three, n)})",
        f"  total positive pairs: {positive:,}",
    ]


def build_report(data_dir: str | Path = "data") -> str:
    data_dir = Path(data_dir)
    train_paths = _require_files(data_dir / "train", TRAIN_FILES)
    test_paths = _require_files(data_dir / "test", TEST_FILES)
    train = {k: _scan_source(v) for k, v in train_paths.items() if k != "ground_truth"}
    test = {k: _scan_source(v) for k, v in test_paths.items()}
    gt = _scan_gt(train_paths["ground_truth"])

    lines = [
        "=" * 68,
        "BUSINESS ENTITY RESOLUTION — MEMORY-CONSCIOUS DATASET AUDIT",
        "=" * 68,
        "",
        "TRAIN SOURCES",
        "-" * 68,
    ]
    for key in ("source1", "source2", "source3"):
        lines.extend(_source_stats(key.upper(), train[key]))
    lines.extend(["", "TEST SOURCES", "-" * 68])
    for key in ("source1", "source2", "source3"):
        lines.extend(_source_stats(key.upper(), test[key]))
    lines.extend(
        ["", "GROUND TRUTH", "-" * 68,
         *_ground_truth_stats(train["source1"], train["source2"], train["source3"], gt),
         "",
         "Audit complete. No raw business records are included in this report.",
         "Next stage: partitioned normalization and candidate-recall benchmarking."]
    )
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    print(report)
    output = Path("artifacts") / "eda_report.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved aggregate report to: {output}")


if __name__ == "__main__":
    main()

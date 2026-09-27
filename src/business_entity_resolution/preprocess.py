"""Stage 4A: reusable, validated Parquet preprocessing."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import polars as pl

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRAIN_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
}


def _scan_tsv(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(path, separator="\t", infer_schema=False, low_memory=True)
    missing = [c for c in SOURCE_COLUMNS if c not in lf.collect_schema().names()]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    return lf.select(pl.col(c).cast(pl.String).fill_null("").alias(c) for c in SOURCE_COLUMNS)


def _norm(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalized(lf: pl.LazyFrame) -> pl.LazyFrame:
    name = _norm("business_name")
    address = _norm("business_address")
    return lf.select(
        "entity_id",
        "business_name",
        "business_address",
        pl.col("country").str.strip_chars().alias("country"),
        name.alias("name_norm"),
        name.str.replace_all(" ", "").alias("name_compact"),
        address.alias("address_norm"),
        address.str.extract_all(r"\d+").list.join("|").alias("address_numbers"),
    )


def _counts(lf: pl.LazyFrame) -> dict:
    row = lf.select(
        pl.len().alias("rows"),
        pl.col("entity_id").n_unique().alias("unique_ids"),
    ).collect(engine="streaming").row(0, named=True)
    countries = (
        lf.group_by("country").len().sort("country")
        .collect(engine="streaming")
        .iter_rows(named=True)
    )
    return {
        "rows": int(row["rows"]),
        "unique_ids": int(row["unique_ids"]),
        "countries": {r["country"]: int(r["len"]) for r in countries},
    }


def preprocess_source(input_path: Path, output_root: Path, source: str) -> dict:
    raw = _scan_tsv(input_path)
    before = _counts(raw)
    if before["rows"] != before["unique_ids"]:
        raise ValueError(f"{source}: duplicate entity_id values detected")

    source_dir = output_root / source
    source_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # Country partitions keep later retrieval bounded. Each country is sunk
    # independently so the complete dataset is never collected in Python.
    for country in before["countries"]:
        dest = source_dir / f"country={country}" / "records.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        normalized(raw).filter(pl.col("country") == country).sink_parquet(
            dest, compression="zstd"
        )

    parquet = pl.scan_parquet(str(source_dir / "country=*" / "records.parquet"))
    after = _counts(parquet)
    if before != after:
        raise RuntimeError(f"{source}: validation failed: input={before}, output={after}")

    return {
        **after,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "partitions": len(after["countries"]),
    }


def build_preprocessed(
    data_dir: str | Path = "data",
    output_dir: str | Path = "artifacts/preprocessed/train",
    *,
    replace: bool = True,
) -> dict:
    data_root = Path(data_dir) / "train"
    out = Path(output_dir)
    missing = [str(data_root / f) for f in TRAIN_FILES.values() if not (data_root / f).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    if out.exists():
        if not replace:
            raise FileExistsError(f"{out} already exists")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    metadata = {"sources": {}}
    for source, filename in TRAIN_FILES.items():
        print(f"Preprocessing {source}...")
        metadata["sources"][source] = preprocess_source(data_root / filename, out, source)

    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    result = build_preprocessed()
    print("=" * 68)
    print("STAGE 4A — PREPROCESSING COMPLETE")
    for source, stats in result["sources"].items():
        print(
            f"{source}: {stats['rows']:,} rows, {stats['partitions']} partitions, "
            f"{stats['elapsed_seconds']:.1f}s"
        )
    print("Validated row counts, country counts, and unique entity IDs.")
    print("Saved under artifacts/preprocessed/train/")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd
import polars as pl

from business_entity_resolution.preprocess import build_preprocessed


def _write(path: Path, rows: list[list[str]]) -> None:
    pd.DataFrame(
        rows, columns=["entity_id", "business_name", "business_address", "country"]
    ).to_csv(path, sep="\t", index=False)


def test_preprocess_partitions_and_validates(tmp_path: Path):
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    _write(train / "train_source1.tsv", [
        ["a", "A.B.C. Ltd", "Plot 14, Main Rd", "US"],
        ["b", "Café Paris", "10 Rue Test", "France"],
    ])
    _write(train / "train_source2.tsv", [["c", "ABC Ltd", "", "US"]])
    _write(train / "train_source3.tsv", [["d", "Cafe Paris", "10 Rue Test", "France"]])

    out = tmp_path / "preprocessed"
    result = build_preprocessed(tmp_path / "data", out)

    assert result["sources"]["source1"]["rows"] == 2
    assert result["sources"]["source1"]["countries"] == {"France": 1, "US": 1}
    assert (out / "source1" / "country=US" / "records.parquet").exists()
    assert (out / "metadata.json").exists()

    us = pl.read_parquet(out / "source1" / "country=US" / "records.parquet")
    assert us["name_norm"][0] == "a b c ltd"
    assert us["name_compact"][0] == "abcltd"
    assert us["address_numbers"][0] == "14"


def test_preprocess_rejects_duplicate_ids(tmp_path: Path):
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    duplicate = [["a", "One", "1 Road", "US"], ["a", "Two", "2 Road", "US"]]
    _write(train / "train_source1.tsv", duplicate)
    _write(train / "train_source2.tsv", [["b", "B", "3 Road", "US"]])
    _write(train / "train_source3.tsv", [["c", "C", "4 Road", "US"]])

    try:
        build_preprocessed(tmp_path / "data", tmp_path / "out")
    except ValueError as exc:
        assert "duplicate entity_id" in str(exc)
    else:
        raise AssertionError("duplicate IDs should fail preprocessing")

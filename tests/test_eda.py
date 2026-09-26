from pathlib import Path

import pandas as pd

from business_entity_resolution.eda import build_report


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_build_report(tmp_path: Path):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    cols = ["entity_id", "business_name", "business_address", "country"]
    _write(train / "train_source1.tsv", [dict(zip(cols, ["s1", "Acme Ltd", "12 Main St", "US"]))])
    _write(train / "train_source2.tsv", [dict(zip(cols, ["s2", "ACME LTD.", "12 Main St", "US"]))])
    _write(train / "train_source3.tsv", [dict(zip(cols, ["s3", "Other", "9 Road", "US"]))])
    _write(train / "train_ground_truth.tsv", [{"source1_entity_id": "s1", "matched_entity_ids": "s2"}])
    _write(test / "test_source1.tsv", [dict(zip(cols, ["t1", "X", "A", "FR"]))])
    _write(test / "test_source2.tsv", [dict(zip(cols, ["t2", "Y", "B", "FR"]))])
    _write(test / "test_source3.tsv", [dict(zip(cols, ["t3", "Z", "C", "FR"]))])

    report = build_report(tmp_path)
    assert "total positive pairs: 1" in report
    assert "exact normalized name: 1 (100.00%)" in report
    assert "FR=1" in report

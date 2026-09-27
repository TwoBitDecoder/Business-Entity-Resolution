from pathlib import Path

import pandas as pd

from business_entity_resolution.residual_profiler import build_profile


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_residual_profile(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    cols = ["entity_id", "business_name", "business_address", "country"]
    _write(train / "train_source1.tsv", [
        dict(zip(cols, ["a", "Acme Trading", "Plot 12 Main Road", "US"])),
    ])
    _write(train / "train_source2.tsv", [
        dict(zip(cols, ["b", "Acme Tradng", "12 Other Road", "US"])),
    ])
    _write(train / "train_source3.tsv", [
        dict(zip(cols, ["c", "Unrelated", "99 West", "US"])),
    ])
    _write(train / "train_ground_truth.tsv", [
        {"source1_entity_id": "a", "matched_entity_ids": "b"},
    ])

    result = build_profile(tmp_path)
    assert result["residual_positive_pairs"] == 1
    assert result["rules"]["name_prefix6"]["residual_true_pairs_retrieved"] == 1

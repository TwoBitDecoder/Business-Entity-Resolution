from pathlib import Path

import pandas as pd

from business_entity_resolution.retrieval_benchmark import build_benchmark


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_retrieval_benchmark(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    cols = ["entity_id", "business_name", "business_address", "country"]
    _write(train / "train_source1.tsv", [
        dict(zip(cols, ["s1", "A.C.M.E Ltd", "12 Main St", "US"])),
        dict(zip(cols, ["s4", "Different", "44 West Rd", "US"])),
    ])
    _write(train / "train_source2.tsv", [
        dict(zip(cols, ["s2", "A C M E Ltd", "99 Elsewhere", "US"])),
    ])
    _write(train / "train_source3.tsv", [
        dict(zip(cols, ["s3", "Other Name", "44 West Rd", "US"])),
    ])
    _write(train / "train_ground_truth.tsv", [
        {"source1_entity_id": "s1", "matched_entity_ids": "s2"},
        {"source1_entity_id": "s4", "matched_entity_ids": "s3"},
    ])

    report = build_benchmark(tmp_path)
    assert report["total_positive_pairs"] == 2
    assert report["rules"]["compact_name"]["true_pairs_retrieved"] == 1
    assert report["rules"]["exact_address"]["true_pairs_retrieved"] == 1
    assert report["rules"]["union"]["true_pairs_retrieved"] == 2
    assert report["rules"]["union"]["positive_pair_recall"] == 1.0

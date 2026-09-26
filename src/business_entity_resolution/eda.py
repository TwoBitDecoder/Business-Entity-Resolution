"""Local dataset validation and aggregate EDA.

This module never requires external data. It reports aggregate statistics only.
Run from the repository root:

    python -m business_entity_resolution.eda
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .io import read_ground_truth, read_source
from .normalization import normalize_text

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


def _require_files(root: Path, names: dict[str, str]) -> dict[str, Path]:
    paths = {key: root / name for key, name in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))
    return paths


def _parse_matches(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _pct(n: int, d: int) -> str:
    return "0.00%" if d == 0 else f"{100 * n / d:.2f}%"


def _source_stats(label: str, df: pd.DataFrame) -> list[str]:
    n = len(df)
    duplicate_ids = int(df["entity_id"].duplicated().sum())
    missing_name = int(df["business_name"].str.strip().eq("").sum())
    missing_address = int(df["business_address"].str.strip().eq("").sum())
    missing_country = int(df["country"].str.strip().eq("").sum())
    countries = Counter(df["country"].str.strip().replace("", "<missing>"))
    country_text = ", ".join(f"{k}={v}" for k, v in sorted(countries.items()))
    return [
        f"{label}: {n:,} records",
        f"  duplicate entity_id: {duplicate_ids:,}",
        f"  missing business_name: {missing_name:,} ({_pct(missing_name, n)})",
        f"  missing business_address: {missing_address:,} ({_pct(missing_address, n)})",
        f"  missing country: {missing_country:,} ({_pct(missing_country, n)})",
        f"  countries: {country_text}",
    ]


def _ground_truth_stats(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, gt: pd.DataFrame
) -> tuple[list[str], list[tuple[str, str]]]:
    s1_ids = set(s1["entity_id"])
    candidate_ids = set(s2["entity_id"]) | set(s3["entity_id"])
    rows: list[tuple[str, str]] = []
    multiplicity = Counter()
    invalid_s1 = 0
    invalid_targets = 0
    repeated_targets = 0

    for row in gt.itertuples(index=False):
        sid = row.source1_entity_id
        matches = _parse_matches(row.matched_entity_ids)
        multiplicity[len(matches)] += 1
        if sid not in s1_ids:
            invalid_s1 += 1
        invalid_targets += sum(mid not in candidate_ids for mid in matches)
        repeated_targets += len(matches) - len(set(matches))
        rows.extend((sid, mid) for mid in matches)

    missing_gt_s1 = len(s1_ids - set(gt["source1_entity_id"]))
    duplicate_gt_s1 = int(gt["source1_entity_id"].duplicated().sum())
    n = len(gt)
    lines = [
        f"Ground-truth rows: {n:,}",
        f"  S1 IDs missing from ground truth: {missing_gt_s1:,}",
        f"  duplicate S1 rows in ground truth: {duplicate_gt_s1:,}",
        f"  unknown S1 IDs in ground truth: {invalid_s1:,}",
        f"  unknown matched S2/S3 IDs: {invalid_targets:,}",
        f"  repeated target IDs within rows: {repeated_targets:,}",
        f"  singleton S1 entities: {multiplicity[0]:,} ({_pct(multiplicity[0], n)})",
        f"  exactly 1 match: {multiplicity[1]:,} ({_pct(multiplicity[1], n)})",
        f"  exactly 2 matches: {multiplicity[2]:,} ({_pct(multiplicity[2], n)})",
        f"  3+ matches: {sum(v for k, v in multiplicity.items() if k >= 3):,} "
        f"({_pct(sum(v for k, v in multiplicity.items() if k >= 3), n)})",
        f"  total positive pairs: {len(rows):,}",
    ]
    return lines, rows


def _positive_similarity(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, pairs: list[tuple[str, str]]
) -> list[str]:
    if not pairs:
        return ["No positive pairs available."]

    left = s1.set_index("entity_id")
    right = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")
    valid_pairs = [(a, b) for a, b in pairs if a in left.index and b in right.index]
    exact_name = exact_address = exact_both = country_equal = 0

    for sid, tid in valid_pairs:
        lrow, rrow = left.loc[sid], right.loc[tid]
        name_eq = bool(normalize_text(lrow["business_name"])) and (
            normalize_text(lrow["business_name"]) == normalize_text(rrow["business_name"])
        )
        address_eq = bool(normalize_text(lrow["business_address"])) and (
            normalize_text(lrow["business_address"])
            == normalize_text(rrow["business_address"])
        )
        exact_name += int(name_eq)
        exact_address += int(address_eq)
        exact_both += int(name_eq and address_eq)
        country_equal += int(
            normalize_text(lrow["country"]) == normalize_text(rrow["country"])
        )

    n = len(valid_pairs)
    return [
        f"Valid positive pairs analysed: {n:,}",
        f"  exact normalized name: {exact_name:,} ({_pct(exact_name, n)})",
        f"  exact normalized address: {exact_address:,} ({_pct(exact_address, n)})",
        f"  exact normalized name + address: {exact_both:,} ({_pct(exact_both, n)})",
        f"  same normalized country: {country_equal:,} ({_pct(country_equal, n)})",
    ]


def build_report(data_dir: str | Path = "data") -> str:
    data_dir = Path(data_dir)
    train_paths = _require_files(data_dir / "train", TRAIN_FILES)
    test_paths = _require_files(data_dir / "test", TEST_FILES)

    train = {k: read_source(v) for k, v in train_paths.items() if k != "ground_truth"}
    gt = read_ground_truth(train_paths["ground_truth"])
    test = {k: read_source(v) for k, v in test_paths.items()}

    lines = [
        "=" * 64,
        "BUSINESS ENTITY RESOLUTION — LOCAL DATASET AUDIT",
        "=" * 64,
        "",
        "TRAIN SOURCES",
        "-" * 64,
    ]
    for key in ("source1", "source2", "source3"):
        lines.extend(_source_stats(key.upper(), train[key]))
    lines.extend(["", "TEST SOURCES", "-" * 64])
    for key in ("source1", "source2", "source3"):
        lines.extend(_source_stats(key.upper(), test[key]))

    gt_lines, pairs = _ground_truth_stats(
        train["source1"], train["source2"], train["source3"], gt
    )
    lines.extend(["", "GROUND TRUTH", "-" * 64, *gt_lines])
    lines.extend(
        [
            "",
            "POSITIVE-PAIR EXACT-MATCH BASELINES",
            "-" * 64,
            *_positive_similarity(
                train["source1"], train["source2"], train["source3"], pairs
            ),
            "",
            "Audit complete. No raw business records are included in this report.",
        ]
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

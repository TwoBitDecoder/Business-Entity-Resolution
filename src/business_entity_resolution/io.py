from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(RECORD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[RECORD_COLUMNS].copy()


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    required = ["source1_entity_id", "matched_entity_ids"]
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[required].copy()

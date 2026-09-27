"""LightGBM pair classifier for engineered candidate features."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

FEATURE_COLUMNS = [
    "name_similarity", "address_similarity", "from_name", "from_address",
    "name_ratio", "name_token_set_ratio", "address_ratio",
    "address_token_set_ratio", "name_exact", "name_compact_exact",
    "address_exact", "name_length_ratio", "address_length_ratio",
    "address_number_jaccard", "address_number_any_overlap",
    "query_address_missing", "target_address_missing",
]


def feature_matrix(frame: pl.DataFrame) -> np.ndarray:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame missing columns: {sorted(missing)}")
    return frame.select(FEATURE_COLUMNS).to_numpy().astype(np.float32, copy=False)


def train_pair_classifier(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    seed: int = 42,
) -> lgb.LGBMClassifier:
    """Train a conservative first baseline; threshold selection is separate."""
    for label, frame in (("train", train), ("validation", validation)):
        if "is_match" not in frame.columns:
            raise ValueError(f"{label} missing column: is_match")
        if frame["is_match"].n_unique() < 2:
            raise ValueError(f"{label} must contain positive and negative pairs")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        feature_matrix(train),
        train["is_match"].cast(pl.Int8).to_numpy(),
        eval_X=feature_matrix(validation),
        eval_y=validation["is_match"].cast(pl.Int8).to_numpy(),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def score_pairs(model: lgb.LGBMClassifier, frame: pl.DataFrame) -> pl.DataFrame:
    probabilities = model.predict_proba(feature_matrix(frame))[:, 1]
    return frame.select("s1_id", "target_id").with_columns(
        pl.Series("match_probability", probabilities, dtype=pl.Float64)
    )

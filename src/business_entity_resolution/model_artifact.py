"""Persist/reload the validated pair classifier with reproducibility metadata."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import polars as pl
from .model import FEATURE_COLUMNS,train_pair_classifier
from .training_data import validate_grouped_split

def train_and_save(labelled: pl.DataFrame, output_dir="artifacts/model", *, seed=42, threshold=.09):
    validate_grouped_split(labelled)
    train=labelled.filter(pl.col("split")=="train")
    valid=labelled.filter(pl.col("split")=="validation")
    if train.is_empty() or valid.is_empty(): raise ValueError("train and validation must be non-empty")
    model=train_pair_classifier(train,valid,seed=seed)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    model_path=out/"pair_model.joblib"; joblib.dump(model,model_path)
    meta={"seed":seed,"decision_threshold":threshold,"feature_columns":FEATURE_COLUMNS,
          "train_pairs":train.height,"validation_pairs":valid.height,
          "best_iteration":getattr(model,"best_iteration_",None)}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    return meta

def load_model_metadata(output_dir="artifacts/model"):
    return json.loads((Path(output_dir)/"metadata.json").read_text(encoding="utf-8"))

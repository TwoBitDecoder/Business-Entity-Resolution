"""Submission-format construction and fail-fast validation."""
from __future__ import annotations
from pathlib import Path
import polars as pl

MATCH_COLUMNS=("source1_entity_id","matched_entity_ids")
PAIR_COLUMNS=("source1_entity_id","candidate_entity_id")

def build_matching_results(source1: pl.DataFrame, scored: pl.DataFrame, *, threshold: float) -> pl.DataFrame:
    if not 0 <= threshold <= 1: raise ValueError("threshold must be in [0, 1]")
    required={"s1_id","target_id","match_probability"}
    if required-set(scored.columns): raise ValueError("scored pairs missing required columns")
    selected=(scored.filter(pl.col("match_probability")>=threshold)
              .group_by("s1_id").agg(pl.col("target_id").unique().sort().str.join(",").alias("matched_entity_ids")))
    return (source1.select(pl.col("entity_id").alias("source1_entity_id")).unique()
            .join(selected,left_on="source1_entity_id",right_on="s1_id",how="left")
            .with_columns(pl.col("matched_entity_ids").fill_null(""))
            .select(MATCH_COLUMNS).sort("source1_entity_id"))

def build_candidate_pairs(candidates: pl.DataFrame) -> pl.DataFrame:
    required={"s1_id","target_id"}
    if required-set(candidates.columns): raise ValueError("candidates missing required columns")
    return (candidates.select(pl.col("s1_id").alias("source1_entity_id"),
                              pl.col("target_id").alias("candidate_entity_id"))
            .unique().sort(PAIR_COLUMNS))

def validate_submission(source1: pl.DataFrame, test_targets: pl.DataFrame,
                        matching: pl.DataFrame, candidate_pairs: pl.DataFrame) -> None:
    s1=set(source1["entity_id"].to_list()); targets=set(test_targets["entity_id"].to_list())
    if matching.columns!=list(MATCH_COLUMNS): raise ValueError("matching_results columns invalid")
    if candidate_pairs.columns!=list(PAIR_COLUMNS): raise ValueError("candidate_pairs columns invalid")
    mids=matching["source1_entity_id"].to_list()
    if len(mids)!=len(s1) or set(mids)!=s1 or len(mids)!=len(set(mids)):
        raise ValueError("every Source1 entity must appear exactly once")
    pairs=set(zip(candidate_pairs["source1_entity_id"].to_list(),candidate_pairs["candidate_entity_id"].to_list()))
    if len(pairs)!=candidate_pairs.height: raise ValueError("duplicate candidate pairs")
    if any(a not in s1 or b not in targets for a,b in pairs): raise ValueError("candidate pair contains invalid ID")
    for row in matching.iter_rows(named=True):
        vals=[] if row["matched_entity_ids"]=="" else row["matched_entity_ids"].split(",")
        if len(vals)!=len(set(vals)): raise ValueError("duplicate matched ID")
        for target in vals:
            if target not in targets: raise ValueError("matched ID is not a test target")
            if (row["source1_entity_id"],target) not in pairs: raise ValueError("final match missing from candidate_pairs")

def write_submission(output_dir: str|Path, matching: pl.DataFrame, candidate_pairs: pl.DataFrame) -> None:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    matching.write_csv(out/"matching_results.tsv",separator="\t")
    candidate_pairs.write_csv(out/"candidate_pairs.tsv",separator="\t")

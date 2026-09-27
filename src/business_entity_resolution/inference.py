"""Chunk-safe inference helpers for candidate scoring and submission output."""
from __future__ import annotations
from pathlib import Path
import joblib
import polars as pl
from .features import build_pair_features
from .model import score_pairs
from .submission import build_candidate_pairs,build_matching_results,validate_submission,write_submission

def score_candidate_parts(model_path, candidate_dir, source1, targets) -> tuple[pl.DataFrame,pl.DataFrame]:
    model=joblib.load(model_path)
    scored=[]; candidates=[]
    for path in sorted(Path(candidate_dir).glob("part-*.parquet")):
        c=pl.read_parquet(path)
        candidates.append(c.select("s1_id","target_id"))
        f=build_pair_features(c,source1,targets)
        scored.append(score_pairs(model,f))
    if not candidates:
        raise ValueError("no candidate parts found")
    return pl.concat(scored),pl.concat(candidates)

def build_validated_submission(*,model_path,candidate_dir,source1,targets,
                               output_dir,threshold=.09):
    scored,candidates=score_candidate_parts(model_path,candidate_dir,source1,targets)
    matching=build_matching_results(source1,scored,threshold=threshold)
    pairs=build_candidate_pairs(candidates)
    validate_submission(source1,targets,matching,pairs)
    write_submission(output_dir,matching,pairs)
    return {"source1_rows":source1.height,"candidate_pairs":pairs.height,
            "predicted_matches":sum(0 if x=="" else len(x.split(",")) for x in matching["matched_entity_ids"])}

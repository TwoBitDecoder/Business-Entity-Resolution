"""Chunk-safe inference helpers for candidate scoring and submission output."""
from __future__ import annotations
from pathlib import Path
import json
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


def load_partitioned_records(root, source, countries):
    frames=[]
    for country in countries:
        path=Path(root)/source/f"country={country}"/"records.parquet"
        if not path.is_file(): raise FileNotFoundError(path)
        frames.append(pl.read_parquet(path))
    return pl.concat(frames,how="vertical")

def run_production_inference(*,preprocessed_root="artifacts/preprocessed/test",
                             candidate_root="artifacts/candidates/test",
                             model_dir="artifacts/model",output_dir="output",
                             countries=("France","India","US")):
    root=Path(preprocessed_root)
    source1=load_partitioned_records(root,"source1",countries)
    targets=pl.concat([load_partitioned_records(root,s,countries) for s in ("source2","source3")],how="vertical")
    candidate_frames=[]
    for country in countries:
        parts=sorted((Path(candidate_root)/f"country={country}").glob("part-*.parquet"))
        if not parts: raise ValueError(f"no candidate parts found for {country}")
        candidate_frames.extend(pl.read_parquet(x) for x in parts)
    candidates=pl.concat(candidate_frames,how="vertical")
    features=build_pair_features(candidates,source1,targets)
    model=joblib.load(Path(model_dir)/"pair_model.joblib")
    metadata=json.loads((Path(model_dir)/"metadata.json").read_text(encoding="utf-8"))
    threshold=float(metadata["decision_threshold"])
    scored=score_pairs(model,features)
    matching=build_matching_results(source1,scored,threshold=threshold)
    pairs=build_candidate_pairs(candidates)
    validate_submission(source1,targets,matching,pairs)
    write_submission(output_dir,matching,pairs)
    result={"countries":list(countries),"source1_rows":source1.height,
            "target_rows":targets.height,"candidate_pairs":pairs.height,
            "threshold":threshold,"matching_rows":matching.height}
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    (Path(output_dir)/"metadata.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def validate_production_inputs(*,preprocessed_root="artifacts/preprocessed/test",
                               candidate_root="artifacts/candidates/test",
                               model_dir="artifacts/model",
                               countries=("France","India","US")):
    """Fail fast before expensive production scoring starts."""
    root=Path(preprocessed_root); croot=Path(candidate_root); mroot=Path(model_dir)
    required=[mroot/"pair_model.joblib",mroot/"metadata.json"]
    for country in countries:
        required.extend([
            root/"source1"/f"country={country}"/"records.parquet",
            root/"source2"/f"country={country}"/"records.parquet",
            root/"source3"/f"country={country}"/"records.parquet",
        ])
        parts=sorted((croot/f"country={country}").glob("part-*.parquet"))
        if not parts: raise FileNotFoundError(f"no candidate parts for {country}")
        expected=list(range(len(parts)))
        actual=[int(x.stem.split("-")[1]) for x in parts]
        if actual!=expected: raise RuntimeError(f"candidate part sequence incomplete for {country}")
    missing=[str(x) for x in required if not x.is_file()]
    if missing: raise FileNotFoundError("missing production inputs: "+", ".join(missing))
    metadata=json.loads((mroot/"metadata.json").read_text(encoding="utf-8"))
    if "decision_threshold" not in metadata: raise ValueError("model metadata missing decision_threshold")
    return {"countries":list(countries),"decision_threshold":float(metadata["decision_threshold"])}

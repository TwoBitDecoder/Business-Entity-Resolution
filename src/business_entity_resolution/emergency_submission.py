"""Create a fast, conservative, validator-compatible emergency submission.

Uses exact normalized name+address agreement within country. Intended only as a
last-resort submission while the full candidate/model pipeline is still running.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

def _read(root: Path, source: str, country: str) -> pl.DataFrame:
    return pl.read_parquet(root/source/f"country={country}"/"records.parquet")

def run(root="artifacts/preprocessed/test", output="output"):
    root=Path(root); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    countries=sorted(p.parent.name.split("=",1)[1] for p in (root/"source1").glob("country=*/records.parquet"))
    match_frames=[]; cand_frames=[]
    for country in countries:
        q=_read(root,"source1",country)
        t=pl.concat([_read(root,s,country) for s in ("source2","source3")],how="vertical")
        # Conservative precision-first fallback: require both normalized fields,
        # and never join blank addresses.
        pairs=(q.select(pl.col("entity_id").alias("source1_entity_id"),"name_compact","address_norm")
            .filter((pl.col("name_compact")!="") & (pl.col("address_norm")!=""))
            .join(t.select(pl.col("entity_id").alias("target_id"),"name_compact","address_norm")
                  .filter((pl.col("name_compact")!="") & (pl.col("address_norm")!="")),
                  on=["name_compact","address_norm"],how="inner")
            .select("source1_entity_id","target_id").unique())
        grouped=(pairs.group_by("source1_entity_id")
                 .agg(pl.col("target_id").sort().str.join(",").alias("ids")))
        base=q.select(pl.col("entity_id").alias("source1_entity_id"))
        matches=(base.join(grouped,on="source1_entity_id",how="left")
                 .with_columns(pl.col("ids").fill_null("").str.strip_chars('"').alias("matched_entity_ids"))
                 .select("source1_entity_id","matched_entity_ids"))
        candidates=(base.join(grouped,on="source1_entity_id",how="left")
                    .with_columns(pl.col("ids").fill_null("").str.strip_chars('"').alias("candidate_entity_ids"))
                    .select("source1_entity_id","candidate_entity_ids"))
        match_frames.append(matches); cand_frames.append(candidates)
        print(f"{country}: {q.height:,} S1; {pairs.height:,} exact pairs",flush=True)
    matching=pl.concat(match_frames).sort("source1_entity_id")
    candidates=pl.concat(cand_frames).sort("source1_entity_id")
    matching.write_csv(out/"matching_results.tsv",separator="\t")
    candidates.write_csv(out/"candidate_pairs.tsv",separator="\t")
    print(f"WROTE {matching.height:,} rows to {out}/matching_results.tsv")
    print(f"WROTE {candidates.height:,} rows to {out}/candidate_pairs.tsv")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="artifacts/preprocessed/test")
    p.add_argument("--output",default="output"); a=p.parse_args(); run(a.root,a.output)
if __name__=="__main__": main()

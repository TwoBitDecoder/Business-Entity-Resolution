"""Production hybrid candidate generation with reusable target indexes.

Frozen R5 config: char-TFIDF name Top-20 + address Top-20, <=40 final.
Target TF-IDF is built once per signal/country and reused across query chunks.
"""
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import polars as pl
from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates, validate_hybrid_bound
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex

NAME_K=20
ADDRESS_K=20
FINAL_K=40
FIELDS=["entity_id","name_compact","address_norm"]


def _load_targets(root: Path, country: str) -> pl.DataFrame:
    frames=[]
    for source in ("source2","source3"):
        frames.append(pl.scan_parquet(_partition_path(root,source,country)).select(FIELDS).collect(engine="streaming"))
    return pl.concat(frames,how="vertical")


def generate_country(root: Path, output_root: Path, country: str, *, chunk_size: int=1000) -> dict:
    if chunk_size < 1: raise ValueError("chunk_size must be >= 1")
    qpath=_partition_path(root,"source1",country)
    qrows=pl.scan_parquet(qpath).select(pl.len()).collect(engine="streaming").item()
    print(f"{country}: loading target records...",flush=True)
    targets=_load_targets(root,country)
    print(f"{country}: loaded {targets.height:,} targets; building name index...",flush=True)
    started=time.perf_counter()
    name_index=SparseTopKIndex(targets,text_column="name_compact",
                               config=RetrievalConfig(top_k=NAME_K))
    print(f"{country}: name index ready; building address index...",flush=True)
    address_index=SparseTopKIndex(targets,text_column="address_norm",
                                  config=RetrievalConfig(top_k=ADDRESS_K))
    print(f"{country}: address index ready; processing {qrows:,} queries...",flush=True)
    outdir=output_root/f"country={country}"
    outdir.mkdir(parents=True,exist_ok=True)
    total=0; max_per=0; chunks=0
    for offset in range(0,qrows,chunk_size):
        q=pl.scan_parquet(qpath).select(FIELDS).slice(offset,chunk_size).collect(engine="streaming")
        name=name_index.query(q)
        address=address_index.query(q)
        hybrid=combine_fuzzy_candidates(name,address,final_top_k=FINAL_K)
        validate_hybrid_bound(hybrid,FINAL_K)
        if hybrid.height:
            max_per=max(max_per,int(hybrid.group_by("s1_id").len()["len"].max()))
        hybrid.write_parquet(outdir/f"part-{chunks:06d}.parquet",compression="zstd")
        total+=hybrid.height; chunks+=1
        print(f"{country}: {min(offset+q.height,qrows):,}/{qrows:,} queries; {total:,} candidates",flush=True)
    return {"country":country,"query_rows":qrows,"target_rows":targets.height,
            "candidate_rows":total,"chunks":chunks,"chunk_size":chunk_size,
            "max_candidates_per_query":max_per,"elapsed_seconds":time.perf_counter()-started}


def generate(root="artifacts/preprocessed/train", output="artifacts/candidates/train",
             *, countries=None, chunk_size=1000, replace=True) -> dict:
    root=Path(root); output=Path(output)
    if countries is None:
        countries=sorted(p.parent.name.split("=",1)[1] for p in (root/"source1").glob("country=*/records.parquet"))
    if output.exists() and replace: shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True)
    stats=[generate_country(root,output,c,chunk_size=chunk_size) for c in countries]
    result={"config":{"name_top_k":NAME_K,"address_top_k":ADDRESS_K,"final_top_k":FINAL_K,
                      "chunk_size":chunk_size},"countries":stats}
    (output/"metadata.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="artifacts/preprocessed/train")
    p.add_argument("--output",default="artifacts/candidates/train")
    p.add_argument("--country",action="append",dest="countries")
    p.add_argument("--chunk-size",type=int,default=1000)
    a=p.parse_args()
    result=generate(a.root,a.output,countries=a.countries,chunk_size=a.chunk_size)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()

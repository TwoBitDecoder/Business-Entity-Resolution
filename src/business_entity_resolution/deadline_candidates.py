"""Five-hour submission fallback candidate generator.

Emergency branch only: word (1,2)-gram TF-IDF Name Top-40 + Address Top-40,
merged to <=40 final candidates. This does not replace the validated char-TFIDF
production retriever.
"""
from __future__ import annotations
import argparse, gc, json, math, os, shutil, time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .hybrid_candidates import combine_fuzzy_candidates, validate_hybrid_bound
from .retrieval_benchmark_runner import _partition_path

TOP_K=40
FINAL_K=40


class WordTopKIndex:
    def __init__(self, targets: pl.DataFrame, text_column: str):
        self.text_column=text_column
        t=targets.select("entity_id",text_column).filter(pl.col(text_column)!="")
        self.target_ids=t["entity_id"].to_list()
        self.vectorizer=TfidfVectorizer(analyzer="word",ngram_range=(1,2),
            lowercase=False,dtype=np.float32,norm="l2")
        self.target_matrix_t=self.vectorizer.fit_transform(
            t[text_column].to_list()).T.tocsc()

    def query(self, queries: pl.DataFrame) -> pl.DataFrame:
        q=queries.select("entity_id",self.text_column).filter(pl.col(self.text_column)!="")
        if q.is_empty():
            return pl.DataFrame(schema={"s1_id":pl.String,"target_id":pl.String,
                                        "name_similarity":pl.Float32})
        qm=self.vectorizer.transform(q[self.text_column].to_list()).tocsr()
        sims=sp_matmul_topn(qm,self.target_matrix_t,
            top_n=min(TOP_K,len(self.target_ids)),threshold=0.0,sort=True,n_threads=2).tocsr()
        qids=q["entity_id"].to_list(); rows=[]
        for i,qid in enumerate(qids):
            for j in range(sims.indptr[i],sims.indptr[i+1]):
                rows.append((qid,self.target_ids[int(sims.indices[j])],float(sims.data[j])))
        return pl.DataFrame(rows,schema=["s1_id","target_id","name_similarity"],
                            orient="row").with_columns(pl.col("name_similarity").cast(pl.Float32))


def _targets(root: Path,country: str,column: str)->pl.DataFrame:
    return pl.concat([
        pl.scan_parquet(_partition_path(root,s,country)).select("entity_id",column)
          .collect(engine="streaming") for s in ("source2","source3")
    ],how="vertical")


def _signal(root,qpath,work,country,column,label,qrows,chunk_size):
    work.mkdir(parents=True,exist_ok=True)
    t=_targets(root,country,column); target_rows=t.height
    started=time.perf_counter(); index=WordTopKIndex(t,column)
    index_seconds=time.perf_counter()-started; del t; gc.collect()
    rows=0; resumed=0; chunks=math.ceil(qrows/chunk_size); qs=time.perf_counter()
    for n,offset in enumerate(range(0,qrows,chunk_size)):
        path=work/f"part-{n:06d}.parquet"
        if path.is_file():
            existing=pl.read_parquet(path,columns=["s1_id","target_id"])
            expected=set(pl.scan_parquet(qpath).select("entity_id").slice(offset,chunk_size)
                         .collect(engine="streaming")["entity_id"].to_list())
            if not set(existing["s1_id"].unique().to_list()) <= expected:
                raise RuntimeError(f"stale checkpoint {path}")
            rows+=existing.height; resumed+=1
            print(f"{country} [{label}] resume {n+1}/{chunks}",flush=True); continue
        q=pl.scan_parquet(qpath).select("entity_id",column).slice(offset,chunk_size).collect(engine="streaming")
        cand=index.query(q); cand.write_parquet(path,compression="zstd"); rows+=cand.height
        print(f"{country} [{label}] {n+1}/{chunks} chunks",flush=True)
    elapsed=time.perf_counter()-qs; del index; gc.collect()
    return {"target_rows":target_rows,"candidate_rows":rows,"chunks":chunks,
            "resumed_chunks":resumed,"index_seconds":index_seconds,"query_seconds":elapsed}


def generate_country(root: Path,out: Path,country: str,chunk_size=5000):
    qpath=_partition_path(root,"source1",country)
    qrows=pl.scan_parquet(qpath).select(pl.len()).collect(engine="streaming").item()
    chunks=math.ceil(qrows/chunk_size); work=out/"_work"/f"country={country}"
    name=_signal(root,qpath,work/"name",country,"name_compact","name",qrows,chunk_size)
    addr=_signal(root,qpath,work/"address",country,"address_norm","address",qrows,chunk_size)
    final=out/f"country={country}"; final.mkdir(parents=True,exist_ok=True)
    total=0
    for n in range(chunks):
        fp=final/f"part-{n:06d}.parquet"
        if fp.is_file():
            total+=pl.scan_parquet(fp).select(pl.len()).collect().item(); continue
        a=pl.read_parquet(work/"name"/f"part-{n:06d}.parquet")
        b=pl.read_parquet(work/"address"/f"part-{n:06d}.parquet")
        h=combine_fuzzy_candidates(a,b,final_top_k=FINAL_K); validate_hybrid_bound(h,FINAL_K)
        h.write_parquet(fp,compression="zstd"); total+=h.height
    return {"country":country,"query_rows":qrows,"candidate_rows":total,
            "chunks":chunks,"name_pass":name,"address_pass":addr}


def generate(root="artifacts/preprocessed/test",output="artifacts/candidates/test-deadline",
             countries=None,chunk_size=5000):
    root=Path(root); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if countries is None:
        countries=sorted(p.parent.name.split("=",1)[1] for p in
            (root/"source1").glob("country=*/records.parquet"))
    stats=[generate_country(root,out,c,chunk_size) for c in countries]
    result={"config":{"retriever":"word_1_2","name_top_k":TOP_K,
            "address_top_k":TOP_K,"final_top_k":FINAL_K,"chunk_size":chunk_size},
            "countries":stats}
    (out/"metadata.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="artifacts/preprocessed/test")
    p.add_argument("--output",default="artifacts/candidates/test-deadline")
    p.add_argument("--country",action="append",dest="countries")
    p.add_argument("--chunk-size",type=int,default=5000); a=p.parse_args()
    print(json.dumps(generate(a.root,a.output,a.countries,a.chunk_size),indent=2))


if __name__=="__main__": main()

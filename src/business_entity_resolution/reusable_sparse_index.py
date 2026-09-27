"""Reusable sparse TF-IDF Top-K index.

Fits the validated target representation once, then serves many bounded query
chunks without rebuilding target TF-IDF for every chunk.
"""
from __future__ import annotations
import os
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from .bounded_retrieval import RetrievalConfig


class SparseTopKIndex:
    def __init__(self, targets: pl.DataFrame, *, text_column: str,
                 config: RetrievalConfig | None=None):
        self.cfg=config or RetrievalConfig()
        self.text_column=text_column
        required={"entity_id",text_column}
        missing=required-set(targets.columns)
        if missing: raise ValueError(f"targets missing columns: {sorted(missing)}")
        t=targets.select("entity_id",text_column).filter(pl.col(text_column)!="")
        if t.is_empty(): raise ValueError("targets contain no non-empty retrieval text")
        self.target_ids=t["entity_id"].to_list()
        self.vectorizer=TfidfVectorizer(analyzer="char",
            ngram_range=(self.cfg.ngram_min,self.cfg.ngram_max),lowercase=False,
            dtype=np.float32,norm="l2")
        self.target_matrix=self.vectorizer.fit_transform(t[text_column].to_list()).tocsr()
        self.target_matrix_t=self.target_matrix.T.tocsc()

    def query(self, queries: pl.DataFrame) -> pl.DataFrame:
        required={"entity_id",self.text_column}
        missing=required-set(queries.columns)
        if missing: raise ValueError(f"queries missing columns: {sorted(missing)}")
        q=queries.select("entity_id",self.text_column).filter(pl.col(self.text_column)!="")
        if q.is_empty():
            return pl.DataFrame(schema={"s1_id":pl.String,"target_id":pl.String,
                                        "name_similarity":pl.Float32})
        qm=self.vectorizer.transform(q[self.text_column].to_list()).tocsr()
        n_threads = (os.cpu_count() or 1) if self.cfg.n_jobs == -1 else max(1, self.cfg.n_jobs)
        sims=sp_matmul_topn(qm,self.target_matrix_t,
            top_n=min(self.cfg.top_k,len(self.target_ids)),
            threshold=self.cfg.min_similarity,sort=True,
            n_threads=n_threads).tocsr()
        rows=[]; qids=q["entity_id"].to_list()
        for qi,qid in enumerate(qids):
            start,end=sims.indptr[qi],sims.indptr[qi+1]
            cols=sims.indices[start:end]; scores=sims.data[start:end]
            order=sorted(range(scores.size),
                key=lambda i:(-float(scores[i]),self.target_ids[int(cols[i])]))
            rows.extend((qid,self.target_ids[int(cols[i])],float(scores[i])) for i in order)
        return pl.DataFrame(rows,schema=["s1_id","target_id","name_similarity"],orient="row"
            ).with_columns(pl.col("name_similarity").cast(pl.Float32))

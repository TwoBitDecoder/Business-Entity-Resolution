"""R5.4: batched fuzzy rank and score diagnostic for R5.3 residual truth pairs.

For only the tiny known residual set, compute exact sparse cosine scores and
count how many targets outrank each truth target. This is diagnostic-only and
does not change production retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth
from .sparse_topk_experiment import sparse_matmul_topk


def truth_ranks_batched(
    residuals: pl.DataFrame, targets: pl.DataFrame, *, text_column: str
) -> dict[tuple[str, str], dict]:
    """Fit the target representation once and rank all residual truth pairs."""
    empty = {"score": 0.0, "rank": None, "targets_strictly_above": None}
    if residuals.is_empty():
        return {}

    t = targets.select("entity_id", text_column).filter(pl.col(text_column) != "")
    ids = t["entity_id"].to_list()
    if not ids:
        return {(r["s1_id"], r["target_id"]): empty.copy()
                for r in residuals.iter_rows(named=True)}
    id_to_idx = {tid: i for i, tid in enumerate(ids)}

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=False,
        dtype=np.float32, norm="l2",
    )
    target_matrix = vectorizer.fit_transform(t[text_column].to_list()).tocsr()
    target_t = target_matrix.T.tocsc()

    valid, out = [], {}
    for r in residuals.iter_rows(named=True):
        key = (r["s1_id"], r["target_id"])
        if not r[text_column] or r["target_id"] not in id_to_idx:
            out[key] = empty.copy()
        else:
            valid.append(r)
    if not valid:
        return out

    query_matrix = vectorizer.transform([r[text_column] for r in valid]).tocsr()
    ids_array = np.asarray(ids, dtype=str)
    for qi, r in enumerate(valid):
        scores = (query_matrix[qi] @ target_t).toarray().ravel()
        truth_score = float(scores[id_to_idx[r["target_id"]]])
        above = int(np.count_nonzero(scores > truth_score))
        tied_before = int(np.count_nonzero(
            (scores == truth_score) & (ids_array < r["target_id"])
        ))
        out[(r["s1_id"], r["target_id"])] = {
            "score": truth_score,
            "rank": above + tied_before + 1,
            "targets_strictly_above": above,
        }
    return out

def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact", "address_norm")
        .head(limit)
        .collect(engine="streaming")
    )


def run_rank_diagnostic(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *, country: str = "India", query_count: int = 1_000,
    target_limit_per_source: int = 250_000,
) -> dict:
    base = Path(root)
    queries = _load(_partition_path(base, "source1", country), query_count)
    targets = pl.concat([
        _load(_partition_path(base, "source2", country), target_limit_per_source),
        _load(_partition_path(base, "source3", country), target_limit_per_source),
    ], how="vertical")

    name = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=20), text_column="name_compact"
    )
    address = sparse_matmul_topk(
        queries, targets, config=RetrievalConfig(top_k=10), text_column="address_norm"
    )
    hybrid = combine_fuzzy_candidates(name, address, final_top_k=30)

    truth = load_truth(ground_truth_path, queries["entity_id"].to_list())
    target_ids_frame = targets.select(pl.col("entity_id").alias("target_id")).unique()
    truth_pool = truth.join(target_ids_frame, on="target_id", how="inner")
    misses = truth_pool.join(
        hybrid.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="anti"
    )

    q_fields = queries.select(
        pl.col("entity_id").alias("s1_id"), "name_compact", "address_norm"
    )
    residuals = misses.join(q_fields, on="s1_id")
    name_diag = truth_ranks_batched(residuals, targets, text_column="name_compact")
    address_diag = truth_ranks_batched(residuals, targets, text_column="address_norm")

    rows = []
    for miss in residuals.iter_rows(named=True):
        key = (miss["s1_id"], miss["target_id"])
        n = name_diag[key]
        a = address_diag[key]
        rows.append({
            "s1_id": key[0], "target_id": key[1],
            "name_score": n["score"], "name_rank": n["rank"],
            "name_targets_strictly_above": n["targets_strictly_above"],
            "address_score": a["score"], "address_rank": a["rank"],
            "address_targets_strictly_above": a["targets_strictly_above"],
        })

    return {
        "purpose": "R5.4 batched residual fuzzy rank/similarity diagnostic",
        "country": country,
        "query_rows": queries.height,
        "target_rows": targets.height,
        "residual_pairs": len(rows),
        "diagnostics": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    a = p.parse_args()
    result = run_rank_diagnostic(
        country=a.country, query_count=a.queries,
        target_limit_per_source=a.targets_per_source,
    )
    out = Path("artifacts/retrieval_v2_r5_residual_rank_diagnostic.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

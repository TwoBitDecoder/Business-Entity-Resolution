"""R5.4: diagnose fuzzy rank and score for R5.3 residual truth pairs.

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


def truth_rank_for_signal(
    query_text: str, truth_target_id: str, truth_text: str,
    target_ids: list[str], target_texts: list[str],
) -> dict:
    """Return truth cosine score and deterministic 1-based rank for one signal."""
    if not query_text or not truth_text:
        return {"score": 0.0, "rank": None, "targets_strictly_above": None}

    nonempty = [(tid, txt) for tid, txt in zip(target_ids, target_texts, strict=True) if txt]
    if not nonempty:
        return {"score": 0.0, "rank": None, "targets_strictly_above": None}

    ids = [x[0] for x in nonempty]
    texts = [x[1] for x in nonempty]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=False,
        dtype=np.float32, norm="l2",
    )
    target_matrix = vectorizer.fit_transform(texts).tocsr()
    query_vector = vectorizer.transform([query_text]).tocsr()
    scores = (query_vector @ target_matrix.T).toarray().ravel()
    try:
        truth_idx = ids.index(truth_target_id)
    except ValueError as exc:
        raise ValueError(f"truth target not present: {truth_target_id}") from exc

    truth_score = float(scores[truth_idx])
    strictly_above = int(np.count_nonzero(scores > truth_score))
    # Match the retriever's deterministic presentation tie-break: score desc,
    # target_id asc. This is diagnostic rank, not a change to Top-N selection.
    tied_before = sum(
        1 for i, score in enumerate(scores)
        if float(score) == truth_score and ids[i] < truth_target_id
    )
    return {
        "score": truth_score,
        "rank": strictly_above + tied_before + 1,
        "targets_strictly_above": strictly_above,
    }


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

    target_ids = targets["entity_id"].to_list()
    target_names = targets["name_compact"].to_list()
    target_addresses = targets["address_norm"].to_list()
    q_by_id = {r["entity_id"]: r for r in queries.to_dicts()}
    t_by_id = {r["entity_id"]: r for r in targets.to_dicts()}

    rows = []
    for miss in misses.iter_rows(named=True):
        q = q_by_id[miss["s1_id"]]
        t = t_by_id[miss["target_id"]]
        name_diag = truth_rank_for_signal(
            q["name_compact"], miss["target_id"], t["name_compact"],
            target_ids, target_names,
        )
        address_diag = truth_rank_for_signal(
            q["address_norm"], miss["target_id"], t["address_norm"],
            target_ids, target_addresses,
        )
        rows.append({
            "s1_id": miss["s1_id"], "target_id": miss["target_id"],
            "name_score": name_diag["score"], "name_rank": name_diag["rank"],
            "name_targets_strictly_above": name_diag["targets_strictly_above"],
            "address_score": address_diag["score"], "address_rank": address_diag["rank"],
            "address_targets_strictly_above": address_diag["targets_strictly_above"],
        })

    return {
        "purpose": "R5.4 residual fuzzy rank/similarity diagnostic",
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

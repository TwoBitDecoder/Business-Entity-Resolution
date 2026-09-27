"""R4.1: address-only sparse Top-N correctness and empty-address handling."""

from __future__ import annotations

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk
from .sparse_topk_experiment import sparse_matmul_topk


def compare_address_retrieval(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    top_k: int = 10,
) -> dict:
    """Compare brute and sparse Top-N address retrieval on the same fixture."""
    cfg = RetrievalConfig(top_k=top_k, n_jobs=1)
    baseline = retrieve_topk(
        queries, targets, config=cfg, text_column="address_norm"
    )
    sparse = sparse_matmul_topk(
        queries, targets, config=cfg, text_column="address_norm"
    )
    a = {(r["s1_id"], r["target_id"]): float(r["name_similarity"])
         for r in baseline.iter_rows(named=True)}
    b = {(r["s1_id"], r["target_id"]): float(r["name_similarity"])
         for r in sparse.iter_rows(named=True)}
    common = a.keys() & b.keys()
    return {
        "baseline_pairs": len(a),
        "sparse_topn_pairs": len(b),
        "common_pairs": len(common),
        "baseline_only_pairs": len(a.keys() - b.keys()),
        "sparse_topn_only_pairs": len(b.keys() - a.keys()),
        "max_common_score_abs_delta": max(
            (abs(a[k] - b[k]) for k in common), default=0.0
        ),
        "baseline": baseline,
        "sparse_topn": sparse,
    }

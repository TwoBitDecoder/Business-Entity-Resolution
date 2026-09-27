"""R2.1: diagnose Top-K candidate differences and compare truth recall."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from .r2_name_benchmark import _pair_map
from .retrieval_benchmark_runner import _partition_path
from .retrieval_recall import load_truth, recall_at_k


def _load_ids(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path).select("entity_id").head(limit)
        .collect(engine="streaming")
    )


def _run_child(method: str, country: str, queries: int,
               targets_per_source: int, top_k: int, n_jobs: int) -> dict:
    cmd = [
        sys.executable, "-m", "business_entity_resolution.r2_name_benchmark",
        "--child", method, "--country", country, "--queries", str(queries),
        "--targets-per-source", str(targets_per_source),
        "--top-k", str(top_k), "--n-jobs", str(n_jobs),
    ]
    return json.loads(subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout)


def _candidate_frame(run: dict) -> pl.DataFrame:
    return pl.DataFrame(
        run["pairs"],
        schema=["s1_id", "target_id", "name_similarity"],
        orient="row",
    ).with_columns(pl.col("name_similarity").cast(pl.Float32))


def _difference_diagnostics(a: dict, b: dict) -> dict:
    amap, bmap = _pair_map(a), _pair_map(b)
    a_only, b_only = amap.keys() - bmap.keys(), bmap.keys() - amap.keys()
    affected = {q for q, _ in a_only | b_only}

    # A changed item is a boundary tie/near-tie when its score is effectively
    # equal to the minimum retained score from the other retriever for that query.
    amin: dict[str, float] = {}
    bmin: dict[str, float] = {}
    for q, _, s in a["pairs"]:
        amin[q] = min(amin.get(q, float("inf")), s)
    for q, _, s in b["pairs"]:
        bmin[q] = min(bmin.get(q, float("inf")), s)

    tol = 1e-6
    a_near = sum(abs(amap[k] - bmin[k[0]]) <= tol for k in a_only)
    b_near = sum(abs(bmap[k] - amin[k[0]]) <= tol for k in b_only)
    return {
        "affected_queries": len(affected),
        "baseline_only_pairs": len(a_only),
        "sparse_topn_only_pairs": len(b_only),
        "near_boundary_tolerance": tol,
        "baseline_only_near_sparse_boundary": a_near,
        "sparse_only_near_baseline_boundary": b_near,
        "baseline_only_score_range": [
            min((amap[k] for k in a_only), default=None),
            max((amap[k] for k in a_only), default=None),
        ],
        "sparse_only_score_range": [
            min((bmap[k] for k in b_only), default=None),
            max((bmap[k] for k in b_only), default=None),
        ],
    }


def run_diagnostic(
    root: str | Path = "artifacts/preprocessed/train",
    ground_truth_path: str | Path = "data/train/train_ground_truth.tsv",
    *, country: str = "India", queries: int = 1_000,
    targets_per_source: int = 250_000, top_k: int = 20, n_jobs: int = -1,
) -> dict:
    baseline = _run_child("baseline", country, queries, targets_per_source, top_k, n_jobs)
    sparse = _run_child("sparse_topn", country, queries, targets_per_source, top_k, n_jobs)

    base = Path(root)
    qids = _load_ids(_partition_path(base, "source1", country), queries)
    target_ids = pl.concat([
        _load_ids(_partition_path(base, "source2", country), targets_per_source),
        _load_ids(_partition_path(base, "source3", country), targets_per_source),
    ], how="vertical").rename({"entity_id": "target_id"}).unique()
    truth_all = load_truth(ground_truth_path, qids["entity_id"].to_list())
    truth_pool = truth_all.join(target_ids, on="target_id", how="inner")
    ks = tuple(k for k in (5, 10, 20) if k <= top_k)

    base_frame, sparse_frame = _candidate_frame(baseline), _candidate_frame(sparse)
    base_truth = base_frame.join(truth_pool, on=["s1_id", "target_id"], how="inner").unique().height
    sparse_truth = sparse_frame.join(truth_pool, on=["s1_id", "target_id"], how="inner").unique().height

    return {
        "purpose": "R2.1 candidate-difference and ground-truth diagnostic",
        "settings": {"country": country, "queries": queries,
                     "targets_per_source": targets_per_source,
                     "total_targets": 2 * targets_per_source, "top_k": top_k},
        "difference_diagnostics": _difference_diagnostics(baseline, sparse),
        "truth": {
            "truth_pairs_for_queries": truth_all.height,
            "truth_pairs_in_target_pool": truth_pool.height,
            "baseline_truth_pairs_in_topk": base_truth,
            "sparse_topn_truth_pairs_in_topk": sparse_truth,
            "baseline_recall": recall_at_k(base_frame, truth_pool, ks=ks),
            "sparse_topn_recall": recall_at_k(sparse_frame, truth_pool, ks=ks),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    result = run_diagnostic(
        country=a.country, queries=a.queries,
        targets_per_source=a.targets_per_source, top_k=a.top_k, n_jobs=a.n_jobs,
    )
    out = Path("artifacts/retrieval_v2_r2_difference_diagnostic.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

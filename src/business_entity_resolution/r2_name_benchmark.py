"""R2: controlled name-only benchmark for sparse Top-N vs brute cosine."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig, retrieve_topk, validate_bound
from .retrieval_benchmark_runner import _partition_path
from .sparse_topk_experiment import sparse_matmul_topk


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load(path: Path, limit: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select("entity_id", "name_compact")
        .head(limit)
        .collect(engine="streaming")
    )


def _run_one(method: str, country: str, queries_n: int, targets_per_source: int,
             top_k: int, n_jobs: int) -> dict:
    root = Path("artifacts/preprocessed/train")
    started = time.perf_counter()
    queries = _load(_partition_path(root, "source1", country), queries_n)
    targets = pl.concat([
        _load(_partition_path(root, "source2", country), targets_per_source),
        _load(_partition_path(root, "source3", country), targets_per_source),
    ], how="vertical")
    load_s = time.perf_counter() - started
    cfg = RetrievalConfig(top_k=top_k, n_jobs=n_jobs)
    t = time.perf_counter()
    if method == "baseline":
        candidates = retrieve_topk(queries, targets, config=cfg)
    elif method == "sparse_topn":
        candidates = sparse_matmul_topk(queries, targets, config=cfg)
    else:
        raise ValueError(f"unknown method: {method}")
    retrieval_s = time.perf_counter() - t
    nonempty = queries.filter(pl.col("name_compact") != "").height
    validate_bound(candidates, nonempty, top_k)
    return {
        "method": method, "country": country, "query_rows": queries.height,
        "target_rows": targets.height, "top_k": top_k,
        "candidate_rows": candidates.height,
        "retrieval_seconds": retrieval_s,
        "total_seconds": time.perf_counter() - started,
        "process_max_rss_mb": _max_rss_mb(),
        "pairs": [
            [r["s1_id"], r["target_id"], float(r["name_similarity"])]
            for r in candidates.iter_rows(named=True)
        ],
        "load_seconds": load_s,
    }


def _pair_map(run: dict) -> dict[tuple[str, str], float]:
    return {(a, b): s for a, b, s in run["pairs"]}


def run_comparison(country: str = "India", queries: int = 1_000,
                   targets_per_source: int = 250_000, top_k: int = 20,
                   n_jobs: int = -1) -> dict:
    runs = {}
    for method in ("baseline", "sparse_topn"):
        cmd = [sys.executable, "-m", "business_entity_resolution.r2_name_benchmark", "--child", method,
               "--country", country, "--queries", str(queries),
               "--targets-per-source", str(targets_per_source),
               "--top-k", str(top_k), "--n-jobs", str(n_jobs)]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        runs[method] = json.loads(proc.stdout)

    a, b = _pair_map(runs["baseline"]), _pair_map(runs["sparse_topn"])
    common = a.keys() & b.keys()
    max_score_delta = max((abs(a[k] - b[k]) for k in common), default=0.0)
    result = {
        "purpose": "R2 name-only sparse Top-N comparison",
        "settings": {"country": country, "queries": queries,
                     "targets_per_source": targets_per_source,
                     "total_targets_requested": 2 * targets_per_source,
                     "top_k": top_k},
        "baseline": {k: v for k, v in runs["baseline"].items() if k != "pairs"},
        "sparse_topn": {k: v for k, v in runs["sparse_topn"].items() if k != "pairs"},
        "equivalence": {
            "baseline_pairs": len(a), "sparse_topn_pairs": len(b),
            "common_pairs": len(common),
            "baseline_only_pairs": len(a.keys() - b.keys()),
            "sparse_topn_only_pairs": len(b.keys() - a.keys()),
            "max_common_score_abs_delta": max_score_delta,
        },
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", choices=("baseline", "sparse_topn"))
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=250000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()
    if a.child:
        print(json.dumps(_run_one(a.child, a.country, a.queries,
                                  a.targets_per_source, a.top_k, a.n_jobs)))
        return
    result = run_comparison(a.country, a.queries, a.targets_per_source,
                            a.top_k, a.n_jobs)
    out = Path("artifacts/retrieval_v2_r2_name_benchmark.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

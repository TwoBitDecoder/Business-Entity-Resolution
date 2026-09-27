"""R3: fresh-process scaling ladder for accepted name-only sparse Top-N."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TOTAL_TARGETS = (125_000, 250_000, 500_000, 1_000_000)


def run_scaling_ladder(
    *, country: str = "India", queries: int = 1_000, top_k: int = 20,
    n_jobs: int = -1, total_targets=DEFAULT_TOTAL_TARGETS,
    stop_peak_rss_mb: float = 10_000.0,
) -> dict:
    runs: list[dict] = []
    stopped_early = False
    for total in total_targets:
        if total % 2:
            raise ValueError("total target count must split evenly across source2/source3")
        per_source = total // 2
        cmd = [
            sys.executable, "-m", "business_entity_resolution.r2_name_benchmark",
            "--child", "sparse_topn", "--country", country,
            "--queries", str(queries), "--targets-per-source", str(per_source),
            "--top-k", str(top_k), "--n-jobs", str(n_jobs),
        ]
        child = json.loads(subprocess.run(
            cmd, check=True, capture_output=True, text=True
        ).stdout)
        child.pop("pairs", None)
        runs.append(child)
        if child["process_max_rss_mb"] >= stop_peak_rss_mb:
            stopped_early = True
            break
    return {
        "purpose": "R3 name-only sparse Top-N scaling ladder",
        "settings": {
            "country": country, "queries": queries, "top_k": top_k,
            "requested_total_targets": list(total_targets),
            "stop_peak_rss_mb": stop_peak_rss_mb,
        },
        "stopped_early": stopped_early,
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--stop-peak-rss-mb", type=float, default=10000.0)
    a = p.parse_args()
    result = run_scaling_ladder(
        country=a.country, queries=a.queries, top_k=a.top_k,
        n_jobs=a.n_jobs, stop_peak_rss_mb=a.stop_peak_rss_mb,
    )
    out = Path("artifacts/retrieval_v2_r3_scaling.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

"""R4.3: isolated address-only Sparse Top-N scaling ladder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TOTAL_TARGETS = (250_000, 500_000, 1_000_000)


def run_scaling_ladder(
    *, country: str = "India", queries: int = 1_000, top_k: int = 10,
    total_targets=DEFAULT_TOTAL_TARGETS, stop_peak_rss_mb: float = 10_000.0,
) -> dict:
    """Run each address benchmark size in a fresh process; gate larger runs by RSS."""
    if queries < 1 or top_k < 1 or stop_peak_rss_mb <= 0:
        raise ValueError("queries, top_k, and stop_peak_rss_mb must be positive")

    runs: list[dict] = []
    stopped_early = False
    for total in total_targets:
        if total < 2 or total % 2:
            raise ValueError("each total target count must be positive and even")
        cmd = [
            sys.executable, "-m", "business_entity_resolution.r4_address_benchmark",
            "--country", country, "--queries", str(queries),
            "--targets-per-source", str(total // 2), "--top-k", str(top_k),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # r4_address_benchmark prints pretty JSON followed by a Saved: line.
        payload = proc.stdout.rsplit("\nSaved:", 1)[0].strip()
        run = json.loads(payload)
        actual_targets = int(run["target_rows"])
        run["requested_total_targets"] = total
        run["target_count_complete"] = actual_targets == total
        runs.append(run)

        # Never advance if the requested pool was not actually available, or if
        # the completed run has crossed the process-memory safety ceiling.
        if (not run["target_count_complete"]
                or float(run["process_max_rss_mb"]) >= stop_peak_rss_mb):
            stopped_early = True
            break

    return {
        "purpose": "R4.3 address-only Sparse Top-N scaling ladder",
        "settings": {
            "country": country,
            "queries": queries,
            "top_k": top_k,
            "requested_total_targets": list(total_targets),
            "stop_peak_rss_mb": stop_peak_rss_mb,
            "fresh_process_per_size": True,
        },
        "stopped_early": stopped_early,
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", default="India")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--stop-peak-rss-mb", type=float, default=10000.0)
    a = p.parse_args()
    result = run_scaling_ladder(
        country=a.country, queries=a.queries, top_k=a.top_k,
        stop_peak_rss_mb=a.stop_peak_rss_mb,
    )
    out = Path("artifacts/retrieval_v2_r4_address_scaling.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

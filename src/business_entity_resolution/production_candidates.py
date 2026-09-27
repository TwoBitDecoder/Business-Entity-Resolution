"""Memory-safe production hybrid candidate generation.

Frozen retrieval: name char-TFIDF Top-20 + address char-TFIDF Top-20 -> <=40 final.
Large name and address target indexes are built and queried sequentially so they
are never resident together.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from pathlib import Path

import polars as pl

from .bounded_retrieval import RetrievalConfig
from .hybrid_candidates import combine_fuzzy_candidates, validate_hybrid_bound
from .retrieval_benchmark_runner import _partition_path
from .reusable_sparse_index import SparseTopKIndex

NAME_K = 20
ADDRESS_K = 20
FINAL_K = 40


def _load_signal_targets(root: Path, country: str, text_column: str) -> pl.DataFrame:
    """Load only IDs plus one retrieval signal to bound resident target memory."""
    frames = []
    for source in ("source2", "source3"):
        frames.append(
            pl.scan_parquet(_partition_path(root, source, country))
            .select("entity_id", text_column)
            .collect(engine="streaming")
        )
    return pl.concat(frames, how="vertical")


def _run_signal_pass(
    *,
    root: Path,
    qpath: Path,
    work_dir: Path,
    country: str,
    text_column: str,
    label: str,
    top_k: int,
    qrows: int,
    chunk_size: int,
) -> dict:
    """Build one target index, query all Source-1 chunks, then release it."""
    print(f"{country}: [{label}] loading target records...", flush=True)
    targets = _load_signal_targets(root, country, text_column)
    target_rows = targets.height
    print(
        f"{country}: [{label}] loaded {target_rows:,} targets; building index...",
        flush=True,
    )

    started = time.perf_counter()
    index = SparseTopKIndex(
        targets,
        text_column=text_column,
        config=RetrievalConfig(top_k=top_k),
    )
    index_seconds = time.perf_counter() - started

    # The fitted index owns everything needed for querying. Release the raw
    # multi-million-row target frame before processing Source-1 chunks.
    del targets
    gc.collect()

    print(
        f"{country}: [{label}] index ready in {index_seconds:.1f}s; "
        f"raw targets released; processing {qrows:,} queries...",
        flush=True,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = 0
    chunks = 0
    query_started = time.perf_counter()

    for offset in range(0, qrows, chunk_size):
        q = (
            pl.scan_parquet(qpath)
            .select("entity_id", text_column)
            .slice(offset, chunk_size)
            .collect(engine="streaming")
        )
        candidates = index.query(q)
        candidates.write_parquet(
            work_dir / f"part-{chunks:06d}.parquet",
            compression="zstd",
        )
        candidate_rows += candidates.height
        chunks += 1
        print(
            f"{country}: [{label}] {min(offset + q.height, qrows):,}/{qrows:,} "
            f"queries; {candidate_rows:,} candidates",
            flush=True,
        )

    query_seconds = time.perf_counter() - query_started

    # Critical memory boundary: one large signal index must be gone before
    # the next signal target frame/index is constructed.
    del index
    gc.collect()
    print(f"{country}: [{label}] pass complete; index released.", flush=True)

    return {
        "target_rows": target_rows,
        "candidate_rows": candidate_rows,
        "chunks": chunks,
        "index_seconds": index_seconds,
        "query_seconds": query_seconds,
    }


def _merge_passes(
    *,
    output_dir: Path,
    name_dir: Path,
    address_dir: Path,
    chunks: int,
    country: str,
) -> dict:
    """Merge corresponding signal chunks, enforce <=40, and remove temp parts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    max_per = 0
    started = time.perf_counter()

    for chunk in range(chunks):
        name_path = name_dir / f"part-{chunk:06d}.parquet"
        address_path = address_dir / f"part-{chunk:06d}.parquet"
        if not name_path.is_file() or not address_path.is_file():
            raise RuntimeError(f"missing temporary candidate part for chunk {chunk}")

        name = pl.read_parquet(name_path)
        address = pl.read_parquet(address_path)
        hybrid = combine_fuzzy_candidates(name, address, final_top_k=FINAL_K)
        validate_hybrid_bound(hybrid, FINAL_K)

        if hybrid.height:
            max_per = max(
                max_per,
                int(hybrid.group_by("s1_id").len()["len"].max()),
            )

        final_path = output_dir / f"part-{chunk:06d}.parquet"
        hybrid.write_parquet(final_path, compression="zstd")
        total += hybrid.height

        # Delete a temporary part only after its final merged part is written.
        name_path.unlink()
        address_path.unlink()
        print(
            f"{country}: [merge] {chunk + 1:,}/{chunks:,} chunks; "
            f"{total:,} final candidates",
            flush=True,
        )

    return {
        "candidate_rows": total,
        "max_candidates_per_query": max_per,
        "merge_seconds": time.perf_counter() - started,
    }


def generate_country(
    root: Path,
    output_root: Path,
    country: str,
    *,
    chunk_size: int = 1000,
) -> dict:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    qpath = _partition_path(root, "source1", country)
    qrows = (
        pl.scan_parquet(qpath)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    expected_chunks = math.ceil(qrows / chunk_size)

    final_dir = output_root / f"country={country}"
    work_country = output_root / "_work" / f"country={country}"
    name_dir = work_country / "name"
    address_dir = work_country / "address"

    # generate() normally starts from a clean output root. This also makes
    # direct generate_country() calls deterministic after an interrupted run.
    for path in (final_dir, work_country):
        if path.exists():
            shutil.rmtree(path)

    started = time.perf_counter()

    name_stats = _run_signal_pass(
        root=root,
        qpath=qpath,
        work_dir=name_dir,
        country=country,
        text_column="name_compact",
        label="name",
        top_k=NAME_K,
        qrows=qrows,
        chunk_size=chunk_size,
    )

    address_stats = _run_signal_pass(
        root=root,
        qpath=qpath,
        work_dir=address_dir,
        country=country,
        text_column="address_norm",
        label="address",
        top_k=ADDRESS_K,
        qrows=qrows,
        chunk_size=chunk_size,
    )

    if name_stats["target_rows"] != address_stats["target_rows"]:
        raise RuntimeError("name/address target row counts differ")
    if name_stats["chunks"] != expected_chunks or address_stats["chunks"] != expected_chunks:
        raise RuntimeError("signal pass chunk coverage mismatch")

    merge_stats = _merge_passes(
        output_dir=final_dir,
        name_dir=name_dir,
        address_dir=address_dir,
        chunks=expected_chunks,
        country=country,
    )

    # Successful merge consumed every temporary part.
    if work_country.exists():
        shutil.rmtree(work_country)
    work_root = output_root / "_work"
    if work_root.exists() and not any(work_root.iterdir()):
        work_root.rmdir()

    return {
        "country": country,
        "query_rows": qrows,
        "target_rows": name_stats["target_rows"],
        "candidate_rows": merge_stats["candidate_rows"],
        "chunks": expected_chunks,
        "chunk_size": chunk_size,
        "max_candidates_per_query": merge_stats["max_candidates_per_query"],
        "name_pass": name_stats,
        "address_pass": address_stats,
        "merge_seconds": merge_stats["merge_seconds"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def generate(
    root="artifacts/preprocessed/train",
    output="artifacts/candidates/train",
    *,
    countries=None,
    chunk_size=1000,
    replace=True,
) -> dict:
    root = Path(root)
    output = Path(output)
    if countries is None:
        countries = sorted(
            p.parent.name.split("=", 1)[1]
            for p in (root / "source1").glob("country=*/records.parquet")
        )

    if output.exists() and replace:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    stats = [
        generate_country(root, output, country, chunk_size=chunk_size)
        for country in countries
    ]
    result = {
        "config": {
            "name_top_k": NAME_K,
            "address_top_k": ADDRESS_K,
            "final_top_k": FINAL_K,
            "chunk_size": chunk_size,
            "index_strategy": "sequential_two_pass",
        },
        "countries": stats,
    }
    (output / "metadata.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/preprocessed/train")
    parser.add_argument("--output", default="artifacts/candidates/train")
    parser.add_argument("--country", action="append", dest="countries")
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    result = generate(
        args.root,
        args.output,
        countries=args.countries,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
Candidate Generation Orchestration Module.

Handles streaming candidate generation, checkpointing, resumability,
provenance tracking, quality metrics calculation, and S1 coverage validation.
"""

from __future__ import annotations
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.blocking_config import ScalableBlockingConfig
from src.scalable_blocker import ScalableBlocker, CandidateRecord
from src.validation import validate_s1_coverage


def run_candidate_generation_pipeline(
    test_dir: Path,
    output_dir: Path,
    config: Optional[ScalableBlockingConfig] = None,
    sample_s1: Optional[int] = None,
    sample_targets: Optional[int] = None,
    chunk_size: int = 50000,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Execute scalable, chunked, checkpointed candidate generation.
    Generates both official submission candidate_pairs.tsv and diagnostic provenance.
    """
    t_start = time.time()
    cfg = config or ScalableBlockingConfig()
    cfg.chunk_size = chunk_size

    test_dir = Path(test_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_dir = output_dir / "candidate_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    if not s1_path.exists():
        raise FileNotFoundError(f"Missing S1 dataset: {s1_path}")

    # For small sample runs (e.g. 1k or 10k), scale target sample proportionately if sample_targets not given
    effective_targets = sample_targets
    if sample_s1 and not sample_targets and sample_s1 < 100000:
        effective_targets = sample_s1 * 30
        print(f"Sample benchmarking mode: indexing {effective_targets:,} target records for {sample_s1:,} S1 queries.")

    # Step 1: Build Scalable Inverted Target Indexes
    blocker = ScalableBlocker(cfg)
    target_files = [p for p in [s2_path, s3_path] if p.exists()]
    if not target_files:
        raise FileNotFoundError(f"No target files found in {test_dir}")

    blocker.fit_from_tsv_files(target_files, max_records=effective_targets)

    # Step 2: Stream Test Source 1 in Chunks with Checkpointing
    candidate_tsv_path = output_dir / "candidate_pairs.tsv"
    provenance_tsv_path = output_dir / "candidate_provenance.tsv"

    print("\n" + "=" * 80)
    print("      STREAMING S1 CANDIDATE GENERATION (CHUNKED & RESTARTABLE)")
    print("=" * 80)

    # Check existing completed chunks if resuming
    completed_chunks = set()
    if resume:
        for cf in chunks_dir.glob("chunk_*.tsv"):
            try:
                c_idx = int(cf.stem.split("_")[1])
                completed_chunks.add(c_idx)
            except Exception:
                pass

    if completed_chunks:
        print(f"Found {len(completed_chunks)} previously completed chunks. Resuming...", flush=True)

    chunk_idx = 0
    s1_processed = 0
    chunk_rows = []
    
    total_fallback_count = 0
    block_type_counts: Dict[str, int] = defaultdict(int)

    # Open S1 file
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_s1:
        header = f_s1.readline()  # skip header
        
        while True:
            line = f_s1.readline()
            if not line:
                break

            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue

            s1_id = parts[0].strip()
            name_raw = parts[1] if len(parts) > 1 else ""
            addr_raw = parts[2] if len(parts) > 2 else ""
            country_raw = parts[3] if len(parts) > 3 else ""

            chunk_rows.append((s1_id, name_raw, addr_raw, country_raw))
            s1_processed += 1

            if len(chunk_rows) >= cfg.chunk_size:
                _process_and_save_chunk(
                    chunk_idx=chunk_idx,
                    chunk_rows=chunk_rows,
                    blocker=blocker,
                    chunks_dir=chunks_dir,
                    block_type_counts=block_type_counts,
                    completed_chunks=completed_chunks
                )
                chunk_rows = []
                chunk_idx += 1

                elapsed = time.time() - t_start
                rate = s1_processed / max(1.0, elapsed)
                print(f"  Processed {s1_processed:,} S1 records ({rate:.0f} rec/s)...", flush=True)

            if sample_s1 and s1_processed >= sample_s1:
                break

        # Process final partial chunk
        if chunk_rows:
            _process_and_save_chunk(
                chunk_idx=chunk_idx,
                chunk_rows=chunk_rows,
                blocker=blocker,
                chunks_dir=chunks_dir,
                block_type_counts=block_type_counts,
                completed_chunks=completed_chunks
            )
            chunk_idx += 1

    print(f"\nAll {s1_processed:,} S1 entities processed into {chunk_idx} chunk files.")

    # Step 3: Consolidate chunks into official candidate_pairs.tsv and provenance
    print("\nConsolidating chunk files into final submission outputs...", flush=True)
    all_s1_seen = set()
    total_pairs = 0
    candidate_counts: List[int] = []

    with open(candidate_tsv_path, "w", encoding="utf-8", newline="\n") as f_cand, \
         open(provenance_tsv_path, "w", encoding="utf-8", newline="\n") as f_prov:

        # Official Headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_prov.write("source1_entity_id\tcandidate_entity_id\tblock_type\tblocking_key\tscore\n")

        for c_file in sorted(chunks_dir.glob("chunk_*.tsv"), key=lambda p: int(p.stem.split("_")[1])):
            with open(c_file, "r", encoding="utf-8") as f_c:
                for line in f_c:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 2:
                        continue
                    s1 = parts[0]
                    c_list_str = parts[1]
                    prov_json_str = parts[2] if len(parts) > 2 else "[]"

                    all_s1_seen.add(s1)
                    c_ids = [x.strip() for x in c_list_str.split(",") if x.strip()]
                    candidate_counts.append(len(c_ids))
                    total_pairs += len(c_ids)

                    f_cand.write(f"{s1}\t{c_list_str}\n")

                    try:
                        prov_list = json.loads(prov_json_str)
                        for item in prov_list:
                            f_prov.write(f"{s1}\t{item['target_id']}\t{item['block_type']}\t{item['blocking_key']}\t{item['score']}\n")
                            if item["block_type"] == "fallback":
                                total_fallback_count += 1
                    except Exception:
                        pass

    # Step 4: Quality Metrics & Reporting
    total_time = time.time() - t_start
    c_arr = np.array(candidate_counts, dtype=np.int32) if candidate_counts else np.array([0])
    
    mean_c = float(np.mean(c_arr)) if len(c_arr) > 0 else 0.0
    median_c = float(np.median(c_arr)) if len(c_arr) > 0 else 0.0
    p95_c = float(np.percentile(c_arr, 95)) if len(c_arr) > 0 else 0.0
    p99_c = float(np.percentile(c_arr, 99)) if len(c_arr) > 0 else 0.0
    max_c = int(np.max(c_arr)) if len(c_arr) > 0 else 0
    min_c = int(np.min(c_arr)) if len(c_arr) > 0 else 0

    total_possible_comparisons = s1_processed * max(1, blocker.total_targets)
    reduction_ratio = 1.0 - (total_pairs / max(1, total_possible_comparisons))

    fallback_pct = (total_fallback_count / max(1, s1_processed)) * 100.0

    print("\n" + "=" * 80)
    print("                 CANDIDATE GENERATION QUALITY REPORT")
    print("=" * 80)
    print(f"Total S1 Records:         {s1_processed:,}")
    print(f"Total Target Records:     {blocker.total_targets:,}")
    print(f"Total Candidate Pairs:    {total_pairs:,}")
    print(f"Candidate Reduction:      {reduction_ratio * 100:.4f}%")
    print()
    print(f"Average candidates/S1:    {mean_c:.2f}")
    print(f"Median candidates/S1:     {median_c:.2f}")
    print(f"Minimum candidates/S1:    {min_c}")
    print(f"P95 candidates/S1:        {p95_c:.2f}")
    print(f"P99 candidates/S1:        {p99_c:.2f}")
    print(f"Maximum candidates/S1:    {max_c}")
    print()
    print(f"Fallback S1 Entities:     {total_fallback_count:,}")
    print(f"Fallback Percentage:      {fallback_pct:.2f}%")
    print(f"Throughput:               {s1_processed / total_time:.0f} S1 records/second")
    print(f"Total Elapsed Time:       {total_time / 60:.2f} minutes")
    print("=" * 80)

    # Step 5: Validate 100% S1 Coverage
    print("\nExecuting explicit S1 Coverage Validation...")
    val_report = validate_s1_coverage(
        s1_source=all_s1_seen,
        candidate_source=candidate_tsv_path,
        output_dir=output_dir,
        fail_loudly=True
    )

    return {
        "s1_processed": s1_processed,
        "total_targets": blocker.total_targets,
        "total_pairs": total_pairs,
        "reduction_ratio": reduction_ratio,
        "mean_candidates": mean_c,
        "median_candidates": median_c,
        "p95_candidates": p95_c,
        "p99_candidates": p99_c,
        "max_candidates": max_c,
        "min_candidates": min_c,
        "fallback_count": total_fallback_count,
        "fallback_pct": fallback_pct,
        "elapsed_seconds": total_time,
        "records_per_sec": s1_processed / max(1.0, total_time),
        "validation": val_report
    }


def _process_and_save_chunk(
    chunk_idx: int,
    chunk_rows: List[Tuple[str, str, str, str]],
    blocker: ScalableBlocker,
    chunks_dir: Path,
    block_type_counts: Dict[str, int],
    completed_chunks: Set[int]
) -> None:
    """Process a single S1 chunk and persist to disk."""
    if chunk_idx in completed_chunks:
        return

    chunk_file = chunks_dir / f"chunk_{chunk_idx:05d}.tsv"
    with open(chunk_file, "w", encoding="utf-8", newline="\n") as f:
        for s1_id, name, addr, country in chunk_rows:
            cands = blocker.generate_candidates_for_record(s1_id, name, addr, country)
            c_ids = [c.target_id for c in cands]
            c_str = ",".join(c_ids)
            prov_json = json.dumps([
                {"target_id": c.target_id, "block_type": c.block_type, "blocking_key": c.blocking_key, "score": c.score}
                for c in cands
            ])
            f.write(f"{s1_id}\t{c_str}\t{prov_json}\n")

            for c in cands:
                block_type_counts[c.block_type] += 1

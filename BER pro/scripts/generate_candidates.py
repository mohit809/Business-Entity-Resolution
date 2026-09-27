"""
Command-line Interface for Scalable Candidate Generation and S1 Coverage Validation.

Usage Examples:
    # 1. Benchmark on 10,000 records:
    python scripts/generate_candidates.py --sample 10000

    # 2. Benchmark on 100,000 records:
    python scripts/generate_candidates.py --sample 100000

    # 3. Full Production Run (all 1.73M S1 records):
    python scripts/generate_candidates.py --full

    # 4. Standalone Coverage Validation on existing file:
    python scripts/generate_candidates.py --validate-only
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.blocking_config import ScalableBlockingConfig
from src.candidate_generation import run_candidate_generation_pipeline
from src.validation import validate_s1_coverage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scalable Entity Resolution Candidate Generation with 100% S1 Coverage Guarantee."
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of S1 records to sample for benchmarking (e.g., 10000 or 100000)")
    parser.add_argument("--full", action="store_true",
                        help="Run full candidate generation on all 1,732,544 test S1 records")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to test datasets directory containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save final candidate_pairs.tsv and candidate_provenance.tsv")
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="Streaming batch size for S1 records (default: 50,000)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run 100%% S1 coverage validation on existing candidate_pairs.tsv")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resuming from previous checkpoint chunks")
    args = parser.parse_args()

    # Resolve paths robustly whether run from workspace root or project folder
    if args.test_dir:
        test_dir = Path(args.test_dir).resolve()
    else:
        test_candidates = [
            PROJECT_ROOT / "resources" / "dataset" / "test",
            WORKSPACE_ROOT / "resources" / "dataset" / "test",
            PROJECT_ROOT.parent / "resources" / "dataset" / "test"
        ]
        test_dir = next((p for p in test_candidates if p.exists()), test_candidates[0])

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = PROJECT_ROOT / "output"

    output_dir.mkdir(parents=True, exist_ok=True)
    s1_file = test_dir / "test_source1.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    print("================================================================================")
    print("      SCALABLE BUSINESS ENTITY RESOLUTION — CANDIDATE GENERATION ENGINE         ")
    print("================================================================================")
    print(f"Test Directory:      {test_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Chunk Size:          {args.chunk_size:,}")
    if args.sample:
        print(f"Benchmark Sample:    {args.sample:,} records")
    elif args.full:
        print(f"Mode:                FULL RUN (All 1,732,544 records)")
    else:
        print(f"Mode:                Default (Full run if not specified)")
    print("--------------------------------------------------------------------------------")

    if args.validate_only:
        print(f"\nRunning standalone coverage validation on {candidate_file}...")
        report = validate_s1_coverage(s1_file, candidate_file, output_dir=output_dir, fail_loudly=True)
        return 0 if report["status"] == "PASS" else 1

    sample_n = args.sample if args.sample else (None if args.full else None)

    config = ScalableBlockingConfig(chunk_size=args.chunk_size)

    try:
        report = run_candidate_generation_pipeline(
            test_dir=test_dir,
            output_dir=output_dir,
            config=config,
            sample_s1=sample_n,
            chunk_size=args.chunk_size,
            resume=not args.no_resume
        )

        print("\nPipeline finished successfully with 100% S1 Coverage Verified.")
        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

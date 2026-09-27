"""
Validation layer for Business Entity Resolution candidate generation.

Enforces the critical invariant:
    set(all_s1_ids) <= set(candidate_pairs["s1_id"])
    100% S1 Coverage, zero missing entities.

Fails loudly if any S1 entity is absent or if coverage drops below 100%.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def _extract_s1_ids_from_source(source: Union[pd.DataFrame, Path, str, Set[str], List[str]]) -> Set[str]:
    """Extract set of S1 IDs from DataFrame, file path, set, or list."""
    if isinstance(source, (set, list, tuple)):
        return set(source)
    if isinstance(source, pd.DataFrame):
        for col in ["entity_id", "source1_entity_id", "s1_id"]:
            if col in source.columns:
                return set(source[col].astype(str).str.strip())
        raise ValueError(f"No S1 ID column found in DataFrame. Columns: {list(source.columns)}")
    
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    
    ids = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        col_idx = 0
        for i, h in enumerate(header):
            if h.strip().lower() in ("entity_id", "source1_entity_id", "s1_id"):
                col_idx = i
                break
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > col_idx:
                val = parts[col_idx].strip()
                if val:
                    ids.add(val)
    return ids


def _load_candidate_pairs_stats(
    candidate_data: Union[pd.DataFrame, Path, str]
) -> Tuple[Dict[str, int], int, int, List[str]]:
    """
    Parse candidate pairs data and compute per-S1 candidate counts, total pairs,
    and detect duplicate candidate pairs.
    Returns:
        (counts_per_s1, total_pairs, duplicate_pairs_count, s1_seen_order)
    """
    counts_per_s1: Dict[str, int] = {}
    s1_seen_order: List[str] = []
    total_pairs = 0
    duplicate_pairs = 0
    
    if isinstance(candidate_data, pd.DataFrame):
        df = candidate_data
        cols = [c.lower() for c in df.columns]
        s1_col = next((c for c in df.columns if c.lower() in ("source1_entity_id", "s1_id", "entity_id")), None)
        if not s1_col:
            raise ValueError(f"Missing S1 column in candidate DataFrame: {list(df.columns)}")

        # Format A: Comma-separated candidate list
        cand_list_col = next((c for c in df.columns if c.lower() in ("candidate_entity_ids", "matched_entity_ids")), None)
        if cand_list_col:
            for s1_val, c_str in zip(df[s1_col], df[cand_list_col]):
                s1 = str(s1_val).strip()
                if s1 not in counts_per_s1:
                    s1_seen_order.append(s1)
                items = [x.strip() for x in str(c_str).split(",") if x.strip() and x.strip() != "nan"]
                unique_items = set(items)
                if len(items) != len(unique_items):
                    duplicate_pairs += (len(items) - len(unique_items))
                cnt = len(unique_items)
                counts_per_s1[s1] = cnt
                total_pairs += cnt
            return counts_per_s1, total_pairs, duplicate_pairs, s1_seen_order

        # Format B: Pairwise rows (source1_entity_id, candidate_entity_id)
        cand_col = next((c for c in df.columns if c.lower() in ("candidate_entity_id", "target_id", "matched_entity_id")), None)
        if cand_col:
            seen_pairs = set()
            for s1_val, c_val in zip(df[s1_col], df[cand_col]):
                s1 = str(s1_val).strip()
                c = str(c_val).strip()
                if s1 not in counts_per_s1:
                    counts_per_s1[s1] = 0
                    s1_seen_order.append(s1)
                pair = (s1, c)
                if pair in seen_pairs:
                    duplicate_pairs += 1
                else:
                    seen_pairs.add(pair)
                    counts_per_s1[s1] += 1
                    total_pairs += 1
            return counts_per_s1, total_pairs, duplicate_pairs, s1_seen_order

    # Format from file path: stream without loading entire file in pandas
    path = Path(candidate_data)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return counts_per_s1, 0, 0, s1_seen_order
        header = [h.strip().lower() for h in header_line.rstrip("\r\n").split("\t")]
        
        is_list_format = "candidate_entity_ids" in header or "matched_entity_ids" in header
        s1_idx = 0
        cand_idx = 1
        for i, h in enumerate(header):
            if h in ("source1_entity_id", "s1_id", "entity_id"):
                s1_idx = i
            elif h in ("candidate_entity_ids", "candidate_entity_id", "target_id", "matched_entity_ids"):
                cand_idx = i

        seen_pairs_set = set() if not is_list_format else None

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or len(parts) <= s1_idx:
                continue
            s1 = parts[s1_idx].strip()
            if not s1:
                continue
            
            cand_str = parts[cand_idx].strip() if len(parts) > cand_idx else ""
            
            if is_list_format:
                if s1 not in counts_per_s1:
                    s1_seen_order.append(s1)
                if cand_str:
                    c_list = [x.strip() for x in cand_str.split(",") if x.strip()]
                    u_list = set(c_list)
                    if len(c_list) != len(u_list):
                        duplicate_pairs += (len(c_list) - len(u_list))
                    c_cnt = len(u_list)
                else:
                    c_cnt = 0
                counts_per_s1[s1] = c_cnt
                total_pairs += c_cnt
            else:
                # Pairwise
                if s1 not in counts_per_s1:
                    counts_per_s1[s1] = 0
                    s1_seen_order.append(s1)
                pair = (s1, cand_str)
                if pair in seen_pairs_set:
                    duplicate_pairs += 1
                else:
                    seen_pairs_set.add(pair)
                    counts_per_s1[s1] += 1
                    total_pairs += 1

    return counts_per_s1, total_pairs, duplicate_pairs, s1_seen_order


def validate_s1_coverage(
    s1_source: Union[pd.DataFrame, Path, str, Set[str]],
    candidate_source: Union[pd.DataFrame, Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    fail_loudly: bool = True
) -> Dict[str, Any]:
    """
    Validate that 100% of S1 entities are covered in candidate_pairs.
    Computes rigorous distribution metrics and fails loudly on coverage < 100%.
    """
    s1_ids = _extract_s1_ids_from_source(s1_source)
    counts_map, total_pairs, dup_count, _ = _load_candidate_pairs_stats(candidate_source)

    total_s1 = len(s1_ids)
    covered_s1_ids = set(counts_map.keys())
    missing_s1_ids = s1_ids - covered_s1_ids
    covered_s1_count = len(s1_ids & covered_s1_ids)
    missing_count = len(missing_s1_ids)
    
    coverage_pct = (covered_s1_count / total_s1 * 100.0) if total_s1 > 0 else 0.0

    # Calculate statistics on candidate counts per S1
    counts_array = np.array(list(counts_map.values()), dtype=np.int32) if counts_map else np.array([0])
    
    min_cand = int(np.min(counts_array)) if len(counts_array) > 0 else 0
    max_cand = int(np.max(counts_array)) if len(counts_array) > 0 else 0
    median_cand = float(np.median(counts_array)) if len(counts_array) > 0 else 0.0
    p95_cand = float(np.percentile(counts_array, 95)) if len(counts_array) > 0 else 0.0
    p99_cand = float(np.percentile(counts_array, 99)) if len(counts_array) > 0 else 0.0
    mean_cand = float(np.mean(counts_array)) if len(counts_array) > 0 else 0.0

    is_pass = (missing_count == 0) and (total_s1 > 0)

    # Print Official Output Banner
    print("\n" + "=" * 40)
    print("FINAL CANDIDATE VALIDATION")
    print("=" * 40)
    print(f"S1 entities:             {total_s1:,}")
    print(f"Covered S1 entities:     {covered_s1_count:,}")
    print(f"Missing S1 entities:     {missing_count:,}")
    print(f"Coverage:                {coverage_pct:.2f}%")
    print()
    print(f"Candidate pairs:         {total_pairs:,}")
    print(f"Duplicate candidate rows:{dup_count:,}")
    print(f"Average candidates/S1:   {mean_cand:.2f}")
    print(f"Median candidates/S1:    {median_cand:.2f}")
    print(f"Minimum candidates/S1:   {min_cand}")
    print(f"P95 candidates/S1:       {p95_cand:.2f}")
    print(f"P99 candidates/S1:       {p99_cand:.2f}")
    print(f"Maximum candidates/S1:   {max_cand}")
    print()
    print(f"STATUS: {'PASS' if is_pass else 'FAIL'}")
    print("=" * 40)

    # Handle Missing Entities
    if not is_pass:
        out_folder = Path(output_dir) if output_dir else Path(".")
        missing_file = out_folder / "missing_s1_entities.tsv"
        try:
            with open(missing_file, "w", encoding="utf-8") as f:
                f.write("source1_entity_id\n")
                for mid in sorted(missing_s1_ids):
                    f.write(f"{mid}\n")
            print(f"\n[ALERT] Wrote {len(missing_s1_ids):,} missing S1 IDs to: {missing_file.resolve()}")
        except Exception as e:
            print(f"\n[ALERT] Could not write missing_s1_entities.tsv: {e}")

        if fail_loudly:
            raise AssertionError(
                f"CRITICAL ERROR: S1 coverage is {coverage_pct:.2f}% (< 100.00%). "
                f"{missing_count:,} S1 entities are missing from candidate_pairs! "
                f"Example missing IDs: {sorted(list(missing_s1_ids))[:5]}"
            )

    return {
        "total_s1": total_s1,
        "covered_s1": covered_s1_count,
        "missing_s1": missing_count,
        "coverage_pct": coverage_pct,
        "total_pairs": total_pairs,
        "duplicate_pairs": dup_count,
        "min_candidates": min_cand,
        "max_candidates": max_cand,
        "median_candidates": median_cand,
        "mean_candidates": mean_cand,
        "p95_candidates": p95_cand,
        "p99_candidates": p99_cand,
        "status": "PASS" if is_pass else "FAIL"
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate 100% S1 Coverage in candidate_pairs.")
    parser.add_argument("--s1", required=True, help="Path to test_source1.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--output-dir", default=".", help="Directory to save missing_s1_entities.tsv if failed")
    parser.add_argument("--no-fail", action="store_true", help="Do not raise exception on failure")
    args = parser.parse_args()

    res = validate_s1_coverage(args.s1, args.candidate, output_dir=args.output_dir, fail_loudly=not args.no_fail)
    sys.exit(0 if res["status"] == "PASS" else 1)

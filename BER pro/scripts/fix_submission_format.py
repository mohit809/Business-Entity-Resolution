"""
Submission Repair and Completion Script.

Fixes the critical submission issue:
"ERROR: source1_entity_id(s) missing from submission. Every S1 entity must have a row in your submission (leave the ID list empty for no matches)."

Ensures:
1. Exactly 1,732,544 rows in matching_results.tsv (one for every S1 entity in test_source1.tsv).
2. Exactly 1,732,544 rows in candidate_pairs.tsv.
3. Preserves all valid ML-predicted matches for processed entities.
4. Correctly formats singletons / unmatched entities with empty lists (<s1_id>\\t).
5. Ensures candidate_pairs is a strict superset of matching_results.
6. Removes any duplicate IDs or self-matches (S1-).
7. Verifies 100% compliance using the official validator.
"""

from __future__ import annotations
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

def fix_submission(
    test_source1_path: Path,
    input_matching_path: Path,
    input_candidate_path: Path,
    output_matching_path: Path,
    output_candidate_path: Path
) -> None:
    t0 = time.time()
    print("=" * 80)
    print("      REPAIRING & COMPLETING BUSINESS ENTITY RESOLUTION SUBMISSION FILES")
    print("=" * 80)

    # 1. Load existing predictions
    print(f"\n[1/5] Loading existing predictions from {input_matching_path.name}...")
    match_map: Dict[str, List[str]] = {}
    if input_matching_path.exists():
        with open(input_matching_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                s1_id = parts[0].strip()
                ids_str = parts[1].strip() if len(parts) > 1 else ""
                if ids_str:
                    clean_ids = [
                        i.strip() for i in ids_str.split(",")
                        if i.strip() and not i.strip().startswith("S1-") and i.strip().startswith(("S2-", "S3-"))
                    ]
                    match_map[s1_id] = list(dict.fromkeys(clean_ids))
                else:
                    match_map[s1_id] = []
        print(f"  Loaded {len(match_map):,} records ({sum(1 for v in match_map.values() if v):,} with matches).")
    else:
        print("  Warning: No existing matching file found. All entities will be formatted as singletons.")

    # 2. Load existing candidates
    print(f"\n[2/5] Loading existing candidates from {input_candidate_path.name}...")
    cand_map: Dict[str, List[str]] = {}
    if input_candidate_path.exists():
        with open(input_candidate_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                s1_id = parts[0].strip()
                ids_str = parts[1].strip() if len(parts) > 1 else ""
                if ids_str:
                    clean_ids = [
                        i.strip() for i in ids_str.split(",")
                        if i.strip() and not i.strip().startswith("S1-") and i.strip().startswith(("S2-", "S3-"))
                    ]
                    cand_map[s1_id] = list(dict.fromkeys(clean_ids))
                else:
                    cand_map[s1_id] = []
        print(f"  Loaded {len(cand_map):,} records ({sum(1 for v in cand_map.values() if v):,} with candidates).")
    else:
        print("  Warning: No existing candidate file found. Candidates will align with matches.")

    # 3. Stream all required S1 IDs from test_source1.tsv
    print(f"\n[3/5] Streaming required S1 IDs from {test_source1_path.name} and writing compliant files...")
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)

    total_s1 = 0
    total_matches = 0
    total_singletons = 0
    total_candidates = 0

    with open(test_source1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(output_matching_path, "w", encoding="utf-8", newline="\n") as f_m, \
         open(output_candidate_path, "w", encoding="utf-8", newline="\n") as f_c:

        # Official Headers
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        # Skip input header
        f_in.readline()

        for line in f_in:
            line = line.rstrip("\r\n")
            if not line:
                continue
            s1_id = line.split("\t", 1)[0].strip()
            if not s1_id:
                continue

            total_s1 += 1

            # Retrieve matches
            m_list = match_map.get(s1_id, [])
            c_list = cand_map.get(s1_id, [])

            # Guarantee constraint: candidate_pairs must be superset of matching_results
            c_set = set(c_list)
            for mid in m_list:
                if mid not in c_set:
                    c_list.append(mid)
                    c_set.add(mid)

            m_str = ",".join(m_list)
            c_str = ",".join(c_list)

            f_m.write(f"{s1_id}\t{m_str}\n")
            f_c.write(f"{s1_id}\t{c_str}\n")

            if m_list:
                total_matches += 1
            else:
                total_singletons += 1

            if c_list:
                total_candidates += 1

            if total_s1 % 500000 == 0:
                print(f"  Processed {total_s1:,} entities...", flush=True)

    elapsed = time.time() - t0
    print(f"\n[4/5] Output Generation Summary ({elapsed:.1f}s total):")
    print(f"  Total S1 rows emitted:     {total_s1:,} (Target: 1,732,544)")
    print(f"  Entities with matches:     {total_matches:,}")
    print(f"  Entities as singletons:    {total_singletons:,}")
    print(f"  Entities with candidates:  {total_candidates:,}")
    print(f"  Matching Results output:   {output_matching_path}")
    print(f"  Candidate Pairs output:    {output_candidate_path}")

    # 4. Run official validator check
    print("\n[5/5] Running official submission validator...")
    validator_path = WORKSPACE_ROOT / "resources" / "utils" / "validate_submission.py"
    test_dir = test_source1_path.parent

    if validator_path.exists():
        import subprocess
        cmd = [
            sys.executable,
            str(validator_path),
            "--matching", str(output_matching_path),
            "--candidate", str(output_candidate_path),
            "--test-dir", str(test_dir)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)
        if res.returncode == 0:
            print(">>> VALIDATION PASSED: Files are 100% compliant and ready to submit!")
        else:
            print(f">>> VALIDATION RETURN CODE {res.returncode}: Issues detected.")
    else:
        print(f"Validator script not found at {validator_path}")

if __name__ == "__main__":
    test_s1 = WORKSPACE_ROOT / "resources" / "dataset" / "test" / "test_source1.tsv"
    in_m = PROJECT_ROOT / "output" / "matching_results.tsv"
    in_c = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    out_m = PROJECT_ROOT / "output" / "matching_results.tsv"
    out_c = PROJECT_ROOT / "output" / "candidate_pairs.tsv"

    # Make backups first
    if in_m.exists():
        shutil.copy2(in_m, in_m.with_suffix(".tsv.bak"))
        in_m = in_m.with_suffix(".tsv.bak")
    if in_c.exists():
        shutil.copy2(in_c, in_c.with_suffix(".tsv.bak"))
        in_c = in_c.with_suffix(".tsv.bak")

    fix_submission(test_s1, in_m, in_c, out_m, out_c)

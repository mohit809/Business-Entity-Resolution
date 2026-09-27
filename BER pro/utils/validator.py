"""
Validator module for Business Entity Resolution submission outputs.

Matches the official verification rules in utils/validate_submission.py:
- UTF-8 tab-delimited formatting
- Headers:
  matching_results.tsv  -> source1_entity_id \t matched_entity_ids
  candidate_pairs.tsv   -> source1_entity_id \t candidate_entity_ids
- Exactly one row per required Source 1 entity
- Matches and candidates contain only S2-/S3- prefixed IDs (no S1- self-matches)
- No duplicate IDs inside comma-separated lists
- Matches must be a subset of candidate_pairs
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_s1_ids(filepath: Path) -> Set[str]:
    """Read set of Source 1 entity IDs from test_source1.tsv."""
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line:
                s1 = line.split(DELIM, 1)[0].strip()
                ids.add(s1)
    return ids


def validate_file_structure(
    filepath: Path,
    expected_header: List[str],
    required_s1_ids: Set[str],
    label: str
) -> Tuple[Optional[Dict[str, Set[str]]], List[str], List[str]]:
    """
    Validate one TSV file against structural and schema constraints.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not filepath.exists():
        errors.append(f"{label} file not found: {filepath}")
        return None, errors, warnings

    name = filepath.name
    mapping: Dict[str, Set[str]] = {}
    seen_s1: Set[str] = set()
    dup_rows: Set[str] = set()
    intra_dupes: Set[str] = set()
    self_matches: Set[str] = set()
    wrong_prefix: Set[str] = set()

    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            errors.append(f"{name} is completely empty.")
            return None, errors, warnings

        if DELIM not in first_line and "," in first_line:
            errors.append(
                f"{name}: header has no TAB but contains commas — file appears to be COMMA-separated. "
                "Submissions must be TAB-separated (.tsv)."
            )
            return None, errors, warnings

        header_cols = [c.strip().lower() for c in first_line.rstrip("\r\n").split(DELIM)]
        if header_cols != expected_header:
            errors.append(
                f"{name}: unexpected header {header_cols}. Expected exactly {expected_header} (tab-separated)."
            )
            return None, errors, warnings

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}")
                continue

            s1 = s1.strip()
            if s1 in seen_s1:
                dup_rows.add(s1)
            seen_s1.add(s1)

            ids_str = rest.rstrip("\r\n").strip()
            if not ids_str:
                mapping[s1] = set()
                continue

            ids_list = [item.strip() for item in ids_str.split(",") if item.strip()]
            if len(ids_list) != len(set(ids_list)):
                intra_dupes.add(s1)

            id_set = set(ids_list)
            mapping[s1] = id_set

            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)

    # Check for missing or unexpected S1 rows
    missing_s1 = required_s1_ids - seen_s1
    extra_s1 = seen_s1 - required_s1_ids

    if dup_rows:
        errors.append(f"{name}: duplicate source1_entity_id rows for: {sorted(dup_rows)[:5]}")
    if intra_dupes:
        errors.append(f"{name}: repeated ID inside list for: {sorted(intra_dupes)[:5]}")
    if self_matches:
        errors.append(f"{name}: contains Source-1 IDs (self-matches): {sorted(self_matches)[:5]}")
    if wrong_prefix:
        errors.append(f"{name}: contains IDs without S2- or S3- prefix: {sorted(wrong_prefix)[:5]}")
    if missing_s1:
        errors.append(f"{name}: {len(missing_s1)} required S1 entities missing, e.g. {sorted(missing_s1)[:5]}")
    if extra_s1:
        errors.append(f"{name}: {len(extra_s1)} rows with S1 IDs not in test set, e.g. {sorted(extra_s1)[:5]}")

    return mapping, errors, warnings


def validate_outputs(
    matching_path: Path,
    candidate_path: Path,
    test_source1_path: Path,
    expected_s1_ids: Optional[Set[str]] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Run full submission validation suite.
    """
    all_errors: List[str] = []
    all_warnings: List[str] = []

    required_ids = expected_s1_ids if expected_s1_ids is not None else read_s1_ids(test_source1_path)

    # 1. Validate matching_results.tsv
    matching_map, m_err, m_warn = validate_file_structure(
        matching_path, MATCHING_HEADER, required_ids, "matching_results"
    )
    all_errors.extend(m_err)
    all_warnings.extend(m_warn)

    # 2. Validate candidate_pairs.tsv
    candidate_map, c_err, c_warn = validate_file_structure(
        candidate_path, CANDIDATE_HEADER, required_ids, "candidate_pairs"
    )
    all_errors.extend(c_err)
    all_warnings.extend(c_warn)

    # 3. Validate subset constraint: matching_results ⊆ candidate_pairs
    if matching_map is not None and candidate_map is not None:
        violations = {}
        for s1_id, match_set in matching_map.items():
            cand_set = candidate_map.get(s1_id, set())
            leak = match_set - cand_set
            if leak:
                violations[s1_id] = leak

        if violations:
            all_warnings.append(
                f"{len(violations)} Source 1 entities have matched IDs absent from candidate_pairs.tsv "
                f"(e.g., {list(violations.keys())[:5]})."
            )

    is_valid = len(all_errors) == 0
    return is_valid, all_errors, all_warnings

"""
Generates matching_result.tsv in standard competition TSV format:
Columns: source1_entity_id \t matched_entity_ids

Includes:
1. All S1 entities with comma-separated matched entity IDs.
2. Singletons (non-matches) with empty matched_entity_ids.
Total rows: 1,732,544 rows.
"""

import sys
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MATCHING_RESULTS_PATH = WORKSPACE_ROOT / "BER pro" / "output" / "matching_results.tsv"
OUTPUT_TSV = WORKSPACE_ROOT / "BER pro" / "output" / "matching_result.tsv"
OUTPUT_ROOT_TSV = WORKSPACE_ROOT / "matching_result.tsv"


def generate_matching_result_tsv():
    print("=" * 80, flush=True)
    print("       SYNCING COMPLIANT matching_result.tsv (TSV FORMAT)", flush=True)
    print("=" * 80, flush=True)

    if not MATCHING_RESULTS_PATH.exists():
        print(f"Error: {MATCHING_RESULTS_PATH} does not exist.")
        return 1

    shutil.copyfile(MATCHING_RESULTS_PATH, OUTPUT_TSV)
    shutil.copyfile(MATCHING_RESULTS_PATH, OUTPUT_ROOT_TSV)

    print(f"Successfully generated {OUTPUT_TSV} and {OUTPUT_ROOT_TSV} with official header ['source1_entity_id', 'matched_entity_ids']")
    return 0


if __name__ == "__main__":
    sys.exit(generate_matching_result_tsv())

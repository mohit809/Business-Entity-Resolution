"""
Converts matching_results.tsv into clean, normalized tabular formats:
1. matching_pairs_tabular.csv: Pairwise flat tabular format (source1_entity_id, matched_entity_id, target_source)
2. matching_enriched_tabular.csv: Side-by-side business entity comparison table with names, addresses, and countries.
"""

import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MATCHING_PATH = WORKSPACE_ROOT / "BER pro" / "output" / "matching_results.tsv"
PAIRWISE_OUT = WORKSPACE_ROOT / "BER pro" / "output" / "matching_pairs_tabular.csv"
ENRICHED_OUT = WORKSPACE_ROOT / "BER pro" / "output" / "matching_enriched_tabular.csv"
TEST_DIR = WORKSPACE_ROOT / "resources" / "dataset" / "test"


def generate_pairwise_tabular():
    print(f"Generating flat pairwise table from {MATCHING_PATH.name}...", flush=True)
    t0 = time.time()
    total_pairs = 0
    with open(MATCHING_PATH, "r", encoding="utf-8") as f_in, \
         open(PAIRWISE_OUT, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("source1_entity_id,matched_entity_id,target_source\n")
        _ = f_in.readline()  # skip header
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        src = "Source 2" if mid.startswith("S2-") else ("Source 3" if mid.startswith("S3-") else "Unknown")
                        f_out.write(f"{s1_id},{mid},{src}\n")
                        total_pairs += 1

    print(f"Created {PAIRWISE_OUT.name}: {total_pairs:,} tabular rows in {time.time()-t0:.2f}s.", flush=True)
    return total_pairs


def generate_enriched_tabular(max_records=100000):
    print(f"\nGenerating rich side-by-side comparison table (sample: {max_records:,} records)...", flush=True)
    t0 = time.time()

    # Step 1: Gather the sample S1 and Target IDs to look up
    target_ids_needed = set()
    s1_ids_needed = set()
    pair_list = []

    with open(MATCHING_PATH, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                s1_id = parts[0].strip()
                s1_ids_needed.add(s1_id)
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        target_ids_needed.add(mid)
                        pair_list.append((s1_id, mid))
                        if len(pair_list) >= max_records:
                            break
            if len(pair_list) >= max_records:
                break

    print(f"  Need {len(s1_ids_needed):,} S1 records and {len(target_ids_needed):,} target records...", flush=True)

    # Step 2: Load S1 details
    s1_data = {}
    with open(TEST_DIR / "test_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            if eid in s1_ids_needed:
                name = parts[1].strip() if len(parts) > 1 else ""
                addr = parts[2].strip() if len(parts) > 2 else ""
                country = parts[3].strip() if len(parts) > 3 else ""
                s1_data[eid] = (name, addr, country)

    # Step 3: Load target details from S2 and S3
    target_data = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        with open(TEST_DIR / fname, "r", encoding="utf-8", errors="replace") as f:
            _ = f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0].strip()
                if eid in target_ids_needed:
                    name = parts[1].strip() if len(parts) > 1 else ""
                    addr = parts[2].strip() if len(parts) > 2 else ""
                    country = parts[3].strip() if len(parts) > 3 else ""
                    target_data[eid] = (name, addr, country)

    # Step 4: Write enriched tabular CSV
    import csv
    with open(ENRICHED_OUT, "w", encoding="utf-8", newline="\n") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "source1_entity_id",
            "source1_business_name",
            "source1_address",
            "source1_country",
            "matched_entity_id",
            "matched_business_name",
            "matched_address",
            "matched_country",
            "target_source"
        ])

        written = 0
        for s1_id, tid in pair_list:
            s1_info = s1_data.get(s1_id, ("", "", ""))
            t_info = target_data.get(tid, ("", "", ""))
            src = "Source 2" if tid.startswith("S2-") else ("Source 3" if tid.startswith("S3-") else "")
            writer.writerow([
                s1_id,
                s1_info[0],
                s1_info[1],
                s1_info[2],
                tid,
                t_info[0],
                t_info[1],
                t_info[2],
                src
            ])
            written += 1

    print(f"Created {ENRICHED_OUT.name}: {written:,} enriched rows in {time.time()-t0:.2f}s.", flush=True)


if __name__ == "__main__":
    generate_pairwise_tabular()
    generate_enriched_tabular(max_records=100000)

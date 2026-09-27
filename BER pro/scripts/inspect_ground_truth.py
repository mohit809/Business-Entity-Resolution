import csv
from pathlib import Path

DATA_DIR = Path("resources/dataset/train")

# Read first 5 S1 IDs from ground truth that have matches
gt_samples = []
targets_needed = set()

with open(DATA_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        m_str = r.get("matched_entity_ids", "")
        if m_str and m_str.strip():
            m_list = [x.strip() for x in m_str.split(",") if x.strip()]
            gt_samples.append((r["source1_entity_id"], m_list))
            targets_needed.update(m_list)
            if len(gt_samples) >= 5:
                break

s1_needed = {s1 for s1, _ in gt_samples}
s1_records = {}

with open(DATA_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r["entity_id"] in s1_needed:
            s1_records[r["entity_id"]] = r
        if len(s1_records) >= len(s1_needed):
            break

target_records = {}
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    with open(DATA_DIR / fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in targets_needed:
                target_records[r["entity_id"]] = r
            if len(target_records) >= len(targets_needed):
                break

for s1_id, t_list in gt_samples:
    s1 = s1_records.get(s1_id, {})
    print("=" * 80)
    print(f"S1 ID:      {s1_id}")
    print(f"S1 Name:    {s1.get('business_name')}")
    print(f"S1 Addr:    {s1.get('business_address')}")
    print(f"S1 Country: {s1.get('country')}")
    print("-" * 40)
    for tid in t_list:
        t = target_records.get(tid, {})
        print(f"  MATCH:    {tid}")
        print(f"    Name:   {t.get('business_name')}")
        print(f"    Addr:   {t.get('business_address')}")
        print(f"    Country:{t.get('country')}")

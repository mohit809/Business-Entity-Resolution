"""
Demonstration and preview script for Business Entity Resolution pipeline.

Runs an interactive demonstration showing:
1. Training and threshold tuning on matched entities + singletons
2. Test candidate generation and ML inference
3. Formatted side-by-side comparison of resolved entities
4. Submission compliance validation
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure BER pro is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config.settings import PipelineConfig, BlockingConfig, ModelConfig
from src.normalization import prepare_records
from src.training import train_pipeline
from src.inference import run_inference
from src.features import extract_pair_features
from utils.validator import validate_outputs


def main():
    print("================================================================================")
    print("      BUSINESS ENTITY RESOLUTION PIPELINE — LIVE PREVIEW & DEMONSTRATION        ")
    print("================================================================================")

    data_dir = PROJECT_ROOT.parent / "resources" / "dataset"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    output_dir = PROJECT_ROOT / "output"
    model_dir = PROJECT_ROOT / "artifacts"

    # Step 1: Load a representative slice from train
    print("\n[1/5] Loading representative training records with verified ground truth matches...")
    gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", nrows=500)
    matched_gt = gt_df[gt_df["matched_entity_ids"].notna() & (gt_df["matched_entity_ids"] != "")]
    target_matched_ids = set()
    for ids in matched_gt["matched_entity_ids"].head(100):
        target_matched_ids.update([i.strip() for i in ids.split(",") if i.strip()])

    s1_ids_needed = set(matched_gt["source1_entity_id"].head(100))

    # Read train records matching these IDs
    print(f"Reading target records corresponding to {len(s1_ids_needed)} matched reference entities...")
    s1_full = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", nrows=10000)
    s2_full = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=10000)
    s3_full = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=10000)

    # Prepare datasets
    s1_train = prepare_records(s1_full.head(300), "S1")
    s2_train = prepare_records(s2_full.head(600), "S2")
    s3_train = prepare_records(s3_full.head(600), "S3")

    config = PipelineConfig(
        blocking=BlockingConfig(max_name=8, max_addr=8, max_char=10, max_total=30),
        model=ModelConfig(n_estimators=100, max_depth=4, learning_rate=0.08)
    )

    # Step 2: Train Model & Calibrate Threshold
    print("\n[2/5] Training pairwise XGBoost classifier and optimizing Macro F0.5 threshold...")
    model, threshold, best_f05 = train_pipeline(
        s1_train=s1_train,
        s2_train=s2_train,
        s3_train=s3_train,
        gt_df=gt_df,
        config=config,
        artifact_dir=model_dir
    )
    print(f"-> Model converged with calibrated decision threshold: {threshold:.2f}")

    # Step 3: Run Inference on Test Set
    print("\n[3/5] Running inference on test dataset sample...")
    ts1_raw = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", nrows=150)
    ts2_raw = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", nrows=300)
    ts3_raw = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", nrows=300)

    ts1 = prepare_records(ts1_raw, "S1")
    ts2 = prepare_records(ts2_raw, "S2")
    ts3 = prepare_records(ts3_raw, "S3")

    matching_file, candidate_file = run_inference(
        s1_test=ts1,
        s2_test=ts2,
        s3_test=ts3,
        model=model,
        config=config,
        output_dir=output_dir,
        chunk_size=10000
    )

    # Step 4: Submission Format Verification
    print("\n[4/5] Running submission compliance validation...")
    is_valid, errors, warnings = validate_outputs(
        matching_path=matching_file,
        candidate_path=candidate_file,
        test_source1_path=test_dir / "test_source1.tsv",
        expected_s1_ids=set(ts1["entity_id"])
    )
    for w in warnings:
        print(f"  [WARNING] {w}")
    if errors:
        print("  [ERROR] Validation failed:")
        for e in errors:
            print(f"    - {e}")
        return 1
    print("  [STATUS] PASS — Output format perfectly complies with competition validator rules.")

    # Step 5: Display Entity Resolution Preview Table
    print("\n[5/5] ENTITY RESOLUTION MATCH PREVIEW:")
    print("=" * 110)
    results_df = pd.read_csv(matching_file, sep="\t")
    cand_df = pd.read_csv(candidate_file, sep="\t")

    total_entities = len(results_df)
    matched_entities = results_df[results_df["matched_entity_ids"].notna() & (results_df["matched_entity_ids"] != "")]
    singleton_entities = results_df[results_df["matched_entity_ids"].isna() | (results_df["matched_entity_ids"] == "")]

    print(f"Total Test Entities Evaluated: {total_entities}")
    print(f"Entities with Predicted Matches: {len(matched_entities)}")
    print(f"Singleton Entities (No Match):   {len(singleton_entities)}")
    print("=" * 110)

    # Join with metadata for rich display
    s1_lookup = ts1.set_index("entity_id").to_dict("index")
    target_lookup = pd.concat([ts2, ts3], ignore_index=True).set_index("entity_id").to_dict("index")

    print("\nSample Output Rows (matching_results.tsv):")
    print(results_df.head(10).to_string(index=False))

    print("\nSample Output Rows (candidate_pairs.tsv):")
    print(cand_df.head(10).to_string(index=False))

    # Show details of candidates and matches
    print("\nDetailed Match Analysis (Sample Source 1 Records & Predictions):")
    print("-" * 110)
    preview_count = 0
    for _, row in results_df.head(15).iterrows():
        s1_id = row["source1_entity_id"]
        matches = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        cands = str(cand_df[cand_df["source1_entity_id"] == s1_id]["candidate_entity_ids"].values[0]) if len(cand_df[cand_df["source1_entity_id"] == s1_id]) else ""

        s1_rec = s1_lookup.get(s1_id, {})
        s1_name = s1_rec.get("business_name", "N/A")
        s1_addr = s1_rec.get("business_address", "N/A")
        s1_country = s1_rec.get("country", "N/A")

        cand_list = [c.strip() for c in cands.split(",") if c.strip()]
        match_list = [m.strip() for m in matches.split(",") if m.strip()]

        print(f"\n[Source 1 Entity] ID: {s1_id} | Name: '{s1_name}' | Addr: '{s1_addr}' | Country: {s1_country}")
        print(f"  Blocking Candidates ({len(cand_list)}): {', '.join(cand_list[:5])}{'...' if len(cand_list) > 5 else ''}")
        if match_list:
            print(f"  => PREDICTED MATCHES ({len(match_list)}):")
            for mid in match_list:
                m_rec = target_lookup.get(mid, {})
                feats = extract_pair_features(s1_rec, m_rec)
                print(f"     * Target ID: {mid}")
                print(f"       Name: '{m_rec.get('business_name', 'N/A')}'")
                print(f"       Addr: '{m_rec.get('business_address', 'N/A')}'")
                print(f"       Metrics: Name Ratio={feats['name_ratio']:.2f}, WRatio={feats['name_wratio']:.2f}, Addr Ratio={feats['addr_ratio']:.2f}, Country Exact={feats['country_exact']}")
        else:
            print("  => PREDICTED SINGLETON (No candidate exceeded the calibrated decision threshold)")

    print("\n" + "=" * 110)
    print("                    PREVIEW EXECUTION COMPLETED SUCCESSFULLY!                           ")
    print("================================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

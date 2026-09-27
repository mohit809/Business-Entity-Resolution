"""
Business Entity Resolution — Main CLI Pipeline Entrypoint.

Supports end-to-end training, threshold optimization, inference, and validation.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config.settings import PipelineConfig
from src.normalization import prepare_records
from src.model import EntityResolutionModel
from src.training import train_pipeline
from src.inference import run_inference
from utils.validator import validate_outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scalable Business Entity Resolution Pipeline with XGBoost & Macro F0.5 Calibration."
    )
    parser.add_argument("--train-dir", type=str, default="../resources/dataset/train",
                        help="Path to training data directory containing train_source1/2/3.tsv and train_ground_truth.tsv")
    parser.add_argument("--test-dir", type=str, default="../resources/dataset/test",
                        help="Path to test data directory containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save final TSV outputs (matching_results.tsv, candidate_pairs.tsv)")
    parser.add_argument("--model-dir", type=str, default="artifacts",
                        help="Directory to save/load trained model artifacts")
    parser.add_argument("--mode", choices=["all", "train_only", "infer_only"], default="all",
                        help="Execution mode: 'all' (train and predict), 'train_only', or 'infer_only'")
    parser.add_argument("--sample-train", type=int, default=2000,
                        help="Number of training reference records (default: 2,000 for fast convergence; set 0 for all)")
    parser.add_argument("--sample-test", type=int, default=None,
                        help="Optional cap on number of test records for development/fast iteration")
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="Batch chunk size for candidate featurization and scoring")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Automatically run submission validation checks after inference")
    return parser.parse_args()


def load_dataset_file(filepath: Path, sep: str = "\t", n_rows: int | None = None) -> pd.DataFrame:
    """Safely load TSV file with optional row limit."""
    if not filepath.exists():
        raise FileNotFoundError(f"Required dataset file not found: {filepath}")
    row_info = f" ({n_rows:,} rows)" if n_rows else ""
    print(f"Loading {filepath.name}{row_info}...", flush=True)
    return pd.read_csv(filepath, sep=sep, nrows=n_rows)


def load_training_dataset(
    train_dir: Path,
    sample_train: int | None = 2000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training records with guaranteed positive matches and hard-negative target pool.
    When sample_train is specified, extracts representative S1 entities (positives + singletons)
    and scans S2/S3 to retrieve all true matches plus decoy/background records.
    """
    if sample_train is None or sample_train <= 0:
        s1_raw = load_dataset_file(train_dir / "train_source1.tsv")
        s2_raw = load_dataset_file(train_dir / "train_source2.tsv")
        s3_raw = load_dataset_file(train_dir / "train_source3.tsv")
        gt_raw = load_dataset_file(train_dir / "train_ground_truth.tsv")
        return s1_raw, s2_raw, s3_raw, gt_raw

    print(f"Reading ground truth for representative sampling ({sample_train:,} reference entities)...", flush=True)
    gt_full = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t")
    has_m = gt_full["matched_entity_ids"].notna() & (gt_full["matched_entity_ids"].astype(str).str.strip() != "")
    matched_gt = gt_full[has_m]
    singleton_gt = gt_full[~has_m]

    n_matched = min(int(sample_train * 0.75), len(matched_gt))
    n_single = min(sample_train - n_matched, len(singleton_gt))

    sample_gt = pd.concat([
        matched_gt.head(n_matched),
        singleton_gt.head(n_single)
    ], ignore_index=True)

    s1_ids = set(sample_gt["source1_entity_id"])
    target_s2_ids = set()
    target_s3_ids = set()
    for m_val in sample_gt["matched_entity_ids"].dropna():
        for m in str(m_val).split(","):
            m = m.strip()
            if m.startswith("S2-"):
                target_s2_ids.add(m)
            elif m.startswith("S3-"):
                target_s3_ids.add(m)

    print(f"Selected {len(s1_ids):,} reference entities ({n_matched:,} with matches, {n_single:,} singletons).", flush=True)
    print(f"Target matches required: {len(target_s2_ids):,} S2, {len(target_s3_ids):,} S3.", flush=True)

    def scan_tsv_for_ids(filepath: Path, needed_ids: set, background_limit: int = 1000) -> pd.DataFrame:
        recs = []
        bg_n = 0
        remaining_ids = set(needed_ids)
        with open(filepath, "r", encoding="utf-8") as f:
            cols = f.readline().rstrip("\r\n").split("\t")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts:
                    continue
                eid = parts[0]
                if eid in remaining_ids:
                    recs.append(parts)
                    remaining_ids.remove(eid)
                elif bg_n < background_limit:
                    recs.append(parts)
                    bg_n += 1
                if not remaining_ids and bg_n >= background_limit:
                    break
        return pd.DataFrame(recs, columns=cols)

    print("Streaming Source 1 records...", flush=True)
    s1_raw = scan_tsv_for_ids(train_dir / "train_source1.tsv", s1_ids, background_limit=0)
    print("Streaming Source 2 records (matches + background)...", flush=True)
    s2_raw = scan_tsv_for_ids(train_dir / "train_source2.tsv", target_s2_ids, background_limit=sample_train)
    print("Streaming Source 3 records (matches + background)...", flush=True)
    s3_raw = scan_tsv_for_ids(train_dir / "train_source3.tsv", target_s3_ids, background_limit=sample_train)

    del gt_full
    return s1_raw, s2_raw, s3_raw, sample_gt


def main() -> int:
    import gc
    args = parse_arguments()
    config = PipelineConfig()

    train_dir = Path(args.train_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================", flush=True)
    print("           BUSINESS ENTITY RESOLUTION PIPELINE — OFFLINE ML SYSTEM               ", flush=True)
    print("================================================================================", flush=True)
    print(f"Mode:         {args.mode}", flush=True)
    print(f"Train Cap:    {args.sample_train:,}" if args.sample_train else "Train Cap:    Full unconstrained", flush=True)
    print(f"Test Cap:     {args.sample_test:,}" if args.sample_test else "Test Cap:     Full unconstrained", flush=True)
    print(f"Train Dir:    {train_dir}", flush=True)
    print(f"Test Dir:     {test_dir}", flush=True)
    print(f"Output:       {output_dir}", flush=True)
    print(f"Artifacts:    {model_dir}", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)

    model: EntityResolutionModel | None = None
    model_path = model_dir / "model.joblib"

    # Stage 1: Training & Calibration
    if args.mode in ("all", "train_only"):
        sample_n = None if (args.sample_train is not None and args.sample_train <= 0) else args.sample_train

        print("\n>>> STAGE 1: INGESTING TRAINING DATA & NORMALIZATION", flush=True)
        s1_raw, s2_raw, s3_raw, gt_raw = load_training_dataset(train_dir, sample_train=sample_n)

        print("Normalizing training records...", flush=True)
        s1_train = prepare_records(s1_raw, "S1")
        s2_train = prepare_records(s2_raw, "S2")
        s3_train = prepare_records(s3_raw, "S3")

        # Explicit cleanup of raw dataframes to free memory
        del s1_raw, s2_raw, s3_raw
        gc.collect()

        print("\n>>> STAGE 2: TRAINING MODEL & TUNING MACRO F0.5 THRESHOLD", flush=True)
        model, best_thresh, best_f05 = train_pipeline(
            s1_train=s1_train,
            s2_train=s2_train,
            s3_train=s3_train,
            gt_df=gt_raw,
            config=config,
            artifact_dir=model_dir
        )
        print(f"Training Complete. Validation Macro F0.5: {best_f05:.6f} | Calibrated Threshold: {best_thresh:.2f}", flush=True)

        # Explicit cleanup of training records
        del s1_train, s2_train, s3_train, gt_raw
        gc.collect()

    # Stage 2: Inference
    if args.mode in ("all", "infer_only"):
        if model is None:
            if not model_path.exists():
                print(f"Error: Model file {model_path} not found for inference mode.", file=sys.stderr, flush=True)
                return 1
            print(f"Loading pretrained model from {model_path}...", flush=True)
            model = EntityResolutionModel.load(model_path)

        print("\n>>> STAGE 3: INGESTING TEST DATA & NORMALIZATION", flush=True)
        target_n_rows = (args.sample_test * 10) if args.sample_test else None
        ts1_raw = load_dataset_file(test_dir / "test_source1.tsv", n_rows=args.sample_test)
        ts2_raw = load_dataset_file(test_dir / "test_source2.tsv", n_rows=target_n_rows)
        ts3_raw = load_dataset_file(test_dir / "test_source3.tsv", n_rows=target_n_rows)

        print("Normalizing test records...", flush=True)
        ts1 = prepare_records(ts1_raw, "S1")
        ts2 = prepare_records(ts2_raw, "S2")
        ts3 = prepare_records(ts3_raw, "S3")

        del ts1_raw, ts2_raw, ts3_raw
        gc.collect()

        print("\n>>> STAGE 4: CANDIDATE GENERATION, ML SCORING & OUTPUT GENERATION")
        matching_file, candidate_file = run_inference(
            s1_test=ts1,
            s2_test=ts2,
            s3_test=ts3,
            model=model,
            config=config,
            output_dir=output_dir,
            chunk_size=args.chunk_size
        )

        # Stage 3: Output Compliance Validation
        if args.validate:
            print("\n>>> STAGE 5: VALIDATING SUBMISSION FORMAT COMPLIANCE")
            s1_test_file = test_dir / "test_source1.tsv"
            is_valid, errors, warnings = validate_outputs(
                matching_path=matching_file,
                candidate_path=candidate_file,
                test_source1_path=s1_test_file,
                expected_s1_ids=set(ts1["entity_id"]) if args.sample_test else None
            )

            for w in warnings:
                print(f"  [WARNING] {w}")
            if errors:
                print("  [FAILED] Submission validation errors detected:")
                for e in errors:
                    print(f"    - {e}")
                return 1
            print("  [SUCCESS] All submission formatting rules verified! Exit code 0 safe to submit.")

    print("\n================================================================================")
    print("                     PIPELINE RUN COMPLETED SUCCESSFULLY                        ")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

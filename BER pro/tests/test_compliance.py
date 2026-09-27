"""
Integration test verifying submission format compliance against validate_submission.py.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from config.settings import PipelineConfig, BlockingConfig, ModelConfig
from src.normalization import prepare_records
from src.training import train_pipeline
from src.inference import run_inference
from utils.validator import validate_outputs


class TestSubmissionCompliance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.train_dir = self.base_path / "train"
        self.test_dir = self.base_path / "test"
        self.output_dir = self.base_path / "output"
        self.model_dir = self.base_path / "artifacts"

        for p in [self.train_dir, self.test_dir, self.output_dir, self.model_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # Synthetic training data
        s1_train = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Widgets", "business_address": "123 Main St, Springfield", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Beta Solar Corp", "business_address": "456 Solar Way, Phoenix", "country": "US"},
            {"entity_id": "S1-3", "business_name": "Global Lone Star", "business_address": "789 Desert Rd, Reno", "country": "US"},
        ])
        s2_train = pd.DataFrame([
            {"entity_id": "S2-10", "business_name": "Acme Widgets Inc", "business_address": "123 Main Street, Springfield", "country": "US"},
            {"entity_id": "S2-20", "business_name": "Random Bakery", "business_address": "1 Baker St, Boston", "country": "US"},
        ])
        s3_train = pd.DataFrame([
            {"entity_id": "S3-30", "business_name": "Beta Solar", "business_address": "456 Solar Way, Phoenix, AZ", "country": "US"},
            {"entity_id": "S3-40", "business_name": "Unrelated Firm", "business_address": "99 Market St, Seattle", "country": "US"},
        ])
        gt_train = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-30"},
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""}, # Singleton
        ])

        s1_train.to_csv(self.train_dir / "train_source1.tsv", sep="\t", index=False)
        s2_train.to_csv(self.train_dir / "train_source2.tsv", sep="\t", index=False)
        s3_train.to_csv(self.train_dir / "train_source3.tsv", sep="\t", index=False)
        gt_train.to_csv(self.train_dir / "train_ground_truth.tsv", sep="\t", index=False)

        # Synthetic test data
        s1_test = pd.DataFrame([
            {"entity_id": "S1-100", "business_name": "Acme Widgets", "business_address": "123 Main St, Springfield", "country": "US"},
            {"entity_id": "S1-200", "business_name": "Beta Solar Solutions", "business_address": "456 Solar Way, Phoenix", "country": "US"},
            {"entity_id": "S1-300", "business_name": "Pure Singleton Business", "business_address": "1000 Isolated Ave, Denver", "country": "US"},
        ])
        s2_test = pd.DataFrame([
            {"entity_id": "S2-500", "business_name": "Acme Widgets Co", "business_address": "123 Main Street, Springfield", "country": "US"},
            {"entity_id": "S2-600", "business_name": "Other Company", "business_address": "50 River St, Austin", "country": "US"},
        ])
        s3_test = pd.DataFrame([
            {"entity_id": "S3-700", "business_name": "Beta Solar Corp", "business_address": "456 Solar Way, Phoenix, AZ", "country": "US"},
            {"entity_id": "S3-800", "business_name": "Another Shop", "business_address": "77 Forest Ave, Portland", "country": "US"},
        ])

        s1_test.to_csv(self.test_dir / "test_source1.tsv", sep="\t", index=False)
        s2_test.to_csv(self.test_dir / "test_source2.tsv", sep="\t", index=False)
        s3_test.to_csv(self.test_dir / "test_source3.tsv", sep="\t", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_compliance(self):
        config = PipelineConfig(
            blocking=BlockingConfig(max_name=5, max_addr=5, max_char=5, max_total=10),
            model=ModelConfig(n_estimators=10, max_depth=3)
        )

        # 1. Normalize
        s1_tr = prepare_records(pd.read_csv(self.train_dir / "train_source1.tsv", sep="\t"), "S1")
        s2_tr = prepare_records(pd.read_csv(self.train_dir / "train_source2.tsv", sep="\t"), "S2")
        s3_tr = prepare_records(pd.read_csv(self.train_dir / "train_source3.tsv", sep="\t"), "S3")
        gt_df = pd.read_csv(self.train_dir / "train_ground_truth.tsv", sep="\t")

        # 2. Train
        model, thresh, f05 = train_pipeline(s1_tr, s2_tr, s3_tr, gt_df, config, self.model_dir)
        self.assertIsNotNone(model)

        # 3. Infer
        s1_te = prepare_records(pd.read_csv(self.test_dir / "test_source1.tsv", sep="\t"), "S1")
        s2_te = prepare_records(pd.read_csv(self.test_dir / "test_source2.tsv", sep="\t"), "S2")
        s3_te = prepare_records(pd.read_csv(self.test_dir / "test_source3.tsv", sep="\t"), "S3")

        matching_file, candidate_file = run_inference(
            s1_test=s1_te,
            s2_test=s2_te,
            s3_test=s3_te,
            model=model,
            config=config,
            output_dir=self.output_dir
        )

        # 4. Check with internal validator
        is_valid, errors, warnings = validate_outputs(
            matching_path=matching_file,
            candidate_path=candidate_file,
            test_source1_path=self.test_dir / "test_source1.tsv"
        )
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")

        # 5. Check with official challenge validator script
        official_validator = Path(__file__).resolve().parents[2] / "resources" / "utils" / "validate_submission.py"
        cmd = [
            sys.executable,
            str(official_validator),
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(self.test_dir),
            "--check-ids"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("Official validator stdout:\n", res.stdout)
        if res.stderr:
            print("Official validator stderr:\n", res.stderr)
        self.assertEqual(res.returncode, 0, f"Official validator returned exit code {res.returncode}")


if __name__ == "__main__":
    unittest.main()

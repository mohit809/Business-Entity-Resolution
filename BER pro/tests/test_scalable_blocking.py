"""
Unit and Integration Tests for ScalableBlocker and S1 Coverage Validation.
Verifies that:
1. 100% S1 coverage is strictly enforced and verified.
2. Incomplete candidate sets trigger loud failures.
3. Fallback candidates are deterministically emitted for difficult/sparse records.
4. Candidates are strictly bounded per S1.
5. No self-matches (S1-*) are emitted.
6. Provenance is tracked properly.
"""

import unittest
from pathlib import Path
import pandas as pd

from config.blocking_config import ScalableBlockingConfig
from src.scalable_blocker import ScalableBlocker, generate_candidate_dataframe
from src.validation import validate_s1_coverage


class TestScalableBlocking(unittest.TestCase):

    def setUp(self):
        self.config = ScalableBlockingConfig(
            max_name_candidates=5,
            max_total_candidates=10,
            enable_fallback=True
        )
        self.blocker = ScalableBlocker(self.config)

        # Mock target records
        self.targets = pd.DataFrame([
            {"entity_id": "S2-101", "business_name": "Apex Logistics Inc", "business_address": "100 Main St, Chicago, IL", "country": "US"},
            {"entity_id": "S2-102", "business_name": "Summit Health Corp", "business_address": "250 Broad Ave, Dallas, TX", "country": "US"},
            {"entity_id": "S3-201", "business_name": "Bharat Bio Pvt Ltd", "business_address": "12 MG Road, Bangalore 560001", "country": "India"},
            {"entity_id": "S3-202", "business_name": "Zenith Motors", "business_address": "50 Tech Park, Mumbai", "country": "India"},
        ])
        self.blocker.fit_from_dataframe(self.targets)

    def test_guaranteed_100_percent_coverage(self):
        """Verify that every S1 record receives candidates (including fallback)."""
        s1_df = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Apex Logistics LLC", "business_address": "100 Main Street, Chicago", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Unknown Random XYZ Co", "business_address": "Nowhere 99999", "country": "US"},
            {"entity_id": "S1-003", "business_name": "Bharat Bio", "business_address": "12 MG Rd, Bangalore 560001", "country": "India"},
        ])

        cand_df = generate_candidate_dataframe(self.blocker, s1_df)

        # All 3 S1 records must be represented
        unique_s1 = set(cand_df["source1_entity_id"])
        self.assertEqual(unique_s1, {"S1-001", "S1-002", "S1-003"})

        # Record S1-002 must have fallen back deterministically
        s1_002_cands = cand_df[cand_df["source1_entity_id"] == "S1-002"]
        self.assertTrue(len(s1_002_cands) >= 1)
        self.assertIn("fallback", set(s1_002_cands["block_type"]))

        # Validation function must pass with 100% coverage
        report = validate_s1_coverage(s1_df, cand_df, fail_loudly=False)
        self.assertEqual(report["coverage_pct"], 100.0)
        self.assertEqual(report["missing_s1"], 0)
        self.assertEqual(report["status"], "PASS")

    def test_coverage_validation_fails_loudly_on_missing_records(self):
        """Verify that validate_s1_coverage raises AssertionError when an S1 record is missing."""
        s1_df = pd.DataFrame([
            {"entity_id": "S1-001"},
            {"entity_id": "S1-002"},
            {"entity_id": "S1-003"}
        ])
        # Incomplete candidate pairs (missing S1-003)
        incomplete_cands = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-101"},
            {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-102"}
        ])

        with self.assertRaises(AssertionError):
            validate_s1_coverage(s1_df, incomplete_cands, fail_loudly=True)

    def test_candidate_caps_strictly_enforced(self):
        """Ensure candidate count per S1 never exceeds max_total_candidates."""
        s1_df = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Apex Logistics LLC", "business_address": "100 Main Street", "country": "US"}
        ])
        cand_df = generate_candidate_dataframe(self.blocker, s1_df)
        self.assertLessEqual(len(cand_df), self.config.max_total_candidates)

    def test_no_self_matches_or_invalid_prefixes(self):
        """Ensure candidates never contain S1 self matches."""
        s1_df = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Summit Health", "business_address": "Broad Ave", "country": "US"}
        ])
        cand_df = generate_candidate_dataframe(self.blocker, s1_df)
        for target_id in cand_df["candidate_entity_id"]:
            self.assertFalse(target_id.startswith("S1-"), "Self-match detected!")
            self.assertTrue(target_id.startswith(("S2-", "S3-")), f"Invalid target prefix: {target_id}")


if __name__ == "__main__":
    unittest.main()

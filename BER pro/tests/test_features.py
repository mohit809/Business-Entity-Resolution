"""
Unit tests for pairwise feature extraction.
"""

import unittest
import pandas as pd
from src.normalization import prepare_records
from src.features import extract_pair_features, build_feature_dataframe, FEATURE_NAMES


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.rec_a = {
            "name_n": "acme global logistics",
            "addr_n": "100 industrial road suite 500",
            "country_n": "us",
            "name_tokens": {"acme", "global", "logistics"},
            "addr_tokens": {"100", "industrial", "road", "suite", "500"},
            "postal_codes": {"10001"}
        }
        self.rec_b = {
            "name_n": "acme global logistics",
            "addr_n": "100 industrial road",
            "country_n": "us",
            "name_tokens": {"acme", "global", "logistics"},
            "addr_tokens": {"100", "industrial", "road"},
            "postal_codes": {"10001"}
        }
        self.rec_c = {
            "name_n": "zenith software",
            "addr_n": "50 tech park bangalore",
            "country_n": "india",
            "name_tokens": {"zenith", "software"},
            "addr_tokens": {"50", "tech", "park", "bangalore"},
            "postal_codes": {"560001"}
        }

    def test_exact_match_features(self):
        f = extract_pair_features(self.rec_a, self.rec_b)
        for name in FEATURE_NAMES:
            self.assertIn(name, f)
        
        self.assertEqual(f["country_exact"], 1.0)
        self.assertAlmostEqual(f["name_ratio"], 1.0, places=2)
        self.assertAlmostEqual(f["name_jaccard"], 1.0, places=2)
        self.assertGreater(f["addr_ratio"], 0.70)
        self.assertEqual(f["same_postal_like"], 1.0)

    def test_mismatch_features(self):
        f = extract_pair_features(self.rec_a, self.rec_c)
        self.assertEqual(f["country_exact"], 0.0)
        self.assertLess(f["name_ratio"], 0.40)
        self.assertEqual(f["name_jaccard"], 0.0)
        self.assertEqual(f["same_postal_like"], 0.0)

    def test_build_feature_dataframe(self):
        s1 = prepare_records(pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Inc", "business_address": "100 Main St", "country": "US"}
        ]), "S1")
        targets = prepare_records(pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Acme Corp", "business_address": "100 Main Street", "country": "US"}
        ]), "S2")
        pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}])

        f_df = build_feature_dataframe(s1, targets, pairs)
        self.assertEqual(len(f_df), 1)
        self.assertIn("name_ratio", f_df.columns)
        self.assertIn("country_exact", f_df.columns)


if __name__ == "__main__":
    unittest.main()

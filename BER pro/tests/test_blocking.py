"""
Unit tests for multi-stage blocking and candidate generation.
"""

import unittest
import pandas as pd
from config.settings import BlockingConfig
from src.normalization import prepare_records
from src.blocking import MultiBlocker


class TestBlocking(unittest.TestCase):
    def setUp(self):
        self.targets = pd.DataFrame([
            {"entity_id": "S2-101", "business_name": "Acme Widgets Inc", "business_address": "100 Industrial Rd, Chicago", "country": "US"},
            {"entity_id": "S2-102", "business_name": "Zenith Dynamics LLC", "business_address": "450 Broadway St, NY", "country": "US"},
            {"entity_id": "S3-201", "business_name": "Acme Industrial Widgets", "business_address": "100 Industrial Road, Chicago, IL", "country": "US"},
            {"entity_id": "S3-202", "business_name": "Bharati Textiles Pvt Ltd", "business_address": "12 MG Road, Bangalore", "country": "India"},
        ])
        self.targets_prep = prepare_records(self.targets, "Target")
        self.config = BlockingConfig(max_name=5, max_addr=5, max_char=5, max_total=10)
        self.blocker = MultiBlocker(self.config).fit(self.targets_prep)

    def test_exact_name_blocking(self):
        query = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Widgets", "business_address": "100 Industrial Rd", "country": "US"}
        ])
        query_prep = prepare_records(query, "S1")
        pairs = self.blocker.generate_candidate_pairs(query_prep)
        
        cand_ids = list(pairs["candidate_entity_id"])
        self.assertIn("S2-101", cand_ids)

    def test_address_token_overlap_blocking(self):
        query = pd.DataFrame([
            {"entity_id": "S1-2", "business_name": "Unknown Name Barber", "business_address": "100 Industrial Rd, Suite 5", "country": "US"}
        ])
        query_prep = prepare_records(query, "S1")
        pairs = self.blocker.generate_candidate_pairs(query_prep)
        
        cand_ids = list(pairs["candidate_entity_id"])
        # Should pick up industrial road records
        self.assertTrue(any(c in cand_ids for c in ["S2-101", "S3-201"]))

    def test_candidate_caps_and_no_self_match(self):
        query = pd.DataFrame([
            {"entity_id": "S1-3", "business_name": "Acme", "business_address": "100 Road", "country": "US"}
        ])
        query_prep = prepare_records(query, "S1")
        pairs = self.blocker.generate_candidate_pairs(query_prep)
        
        self.assertLessEqual(len(pairs), self.config.max_total)
        # Ensure no self match (no S1 candidate)
        for c in pairs["candidate_entity_id"]:
            self.assertFalse(c.startswith("S1-"))
            self.assertTrue(c.startswith(("S2-", "S3-")))


if __name__ == "__main__":
    unittest.main()

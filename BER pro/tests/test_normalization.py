"""
Unit tests for text and domain normalization.
"""

import unittest
import pandas as pd
from src.normalization import (
    norm_text,
    norm_name,
    norm_address,
    norm_country,
    tokenize,
    extract_postal_codes,
    prepare_records,
)


class TestNormalization(unittest.TestCase):
    def test_norm_text_basic(self):
        self.assertEqual(norm_text("Acme & Sons, Corp!"), "acme and sons corp")
        self.assertEqual(norm_text("  Extra   Spaces  "), "extra spaces")
        self.assertEqual(norm_text(None), "")
        self.assertEqual(norm_text(float("nan")), "")

    def test_norm_name_suffix_stripping(self):
        self.assertEqual(norm_name("Acme Logistics Inc."), "acme logistics")
        self.assertEqual(norm_name("Global Tech Private Limited"), "global tech")
        self.assertEqual(norm_name("Alpha Omega LLC"), "alpha omega")
        self.assertEqual(norm_name("Metro Corp"), "metro")

    def test_norm_address_abbreviations(self):
        self.assertEqual(norm_address("123 Main St, Apt 4B"), "123 main street apt 4b")
        self.assertEqual(norm_address("500 Grand Ave, Suite 10"), "500 grand avenue suite 10")
        self.assertEqual(norm_address("10 Highway Rd"), "10 highway road")

    def test_open_set_country(self):
        # Must preserve non-US/India countries without modification or rejection
        self.assertEqual(norm_country("US"), "us")
        self.assertEqual(norm_country("India"), "india")
        self.assertEqual(norm_country("Germany"), "germany")
        self.assertEqual(norm_country("France"), "france")
        self.assertEqual(norm_country("Japan"), "japan")

    def test_extract_postal_codes(self):
        self.assertEqual(extract_postal_codes("New York, NY 10001"), {"10001"})
        self.assertEqual(extract_postal_codes("Bangalore, KA 560001"), {"560001"})
        self.assertEqual(extract_postal_codes("No Zip Code Here"), set())

    def test_prepare_records(self):
        raw_df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Apex Corp", "business_address": "123 Main St", "country": "US"}
        ])
        prepared = prepare_records(raw_df, "S1")
        self.assertIn("name_n", prepared.columns)
        self.assertIn("addr_n", prepared.columns)
        self.assertIn("country_n", prepared.columns)
        self.assertIn("name_tokens", prepared.columns)
        self.assertEqual(prepared.iloc[0]["name_n"], "apex")
        self.assertEqual(prepared.iloc[0]["addr_n"], "123 main street")
        self.assertEqual(prepared.iloc[0]["country_n"], "us")


if __name__ == "__main__":
    unittest.main()

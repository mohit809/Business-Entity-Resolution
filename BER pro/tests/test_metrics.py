"""
Unit tests for Macro F0.5 metric and singleton handling.
"""

import unittest
import numpy as np
from src.metrics import compute_entity_f05, compute_macro_f05_from_predictions, compute_macro_f05_from_arrays


class TestMetrics(unittest.TestCase):
    def test_singleton_correct(self):
        # Both true and predicted empty -> Score is 1.0
        score = compute_entity_f05(set(), set())
        self.assertEqual(score, 1.0)

    def test_singleton_false_positive(self):
        # True is empty (singleton), but model predicted a match -> 0.0
        score = compute_entity_f05(set(), {"S2-100"})
        self.assertEqual(score, 0.0)

    def test_singleton_false_negative(self):
        # True has match, but model predicted singleton -> 0.0
        score = compute_entity_f05({"S2-100"}, set())
        self.assertEqual(score, 0.0)

    def test_precision_weighting_in_f05(self):
        # F0.5 weights precision higher than recall (beta = 0.5)
        # High precision, moderate recall:
        # P = 2/2 = 1.0, R = 2/4 = 0.5
        true_a = {"S2-1", "S2-2", "S2-3", "S2-4"}
        pred_a = {"S2-1", "S2-2"}
        score_high_precision = compute_entity_f05(true_a, pred_a)

        # Moderate precision, high recall:
        # P = 2/4 = 0.5, R = 2/2 = 1.0
        true_b = {"S2-1", "S2-2"}
        pred_b = {"S2-1", "S2-2", "S2-3", "S2-4"}
        score_high_recall = compute_entity_f05(true_b, pred_b)

        self.assertAlmostEqual(score_high_precision, 0.833333, places=4)
        self.assertAlmostEqual(score_high_recall, 0.555555, places=4)
        self.assertGreater(score_high_precision, score_high_recall)

    def test_macro_averaging(self):
        gt_map = {
            "S1-1": {"S2-1"},
            "S1-2": set(), # True singleton
        }
        pred_map = {
            "S1-1": {"S2-1"},
            "S1-2": set(), # Correctly predicted singleton
        }
        macro_score = compute_macro_f05_from_predictions(gt_map, pred_map, ["S1-1", "S1-2"])
        self.assertEqual(macro_score, 1.0)

    def test_array_evaluation(self):
        # 2 groups: S1-1 has 2 candidates (1 pos, 1 neg), S1-2 has 1 candidate (neg)
        y_true = np.array([1, 0, 0])
        y_prob = np.array([0.9, 0.2, 0.1])
        s1_groups = ["S1-1", "S1-1", "S1-2"]
        score = compute_macro_f05_from_arrays(y_true, y_prob, s1_groups, threshold=0.5)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

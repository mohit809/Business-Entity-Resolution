"""
Evaluation metrics for Business Entity Resolution.

Implements entity-level Macro F0.5 evaluation with explicit singleton handling.
"""

from __future__ import annotations
from typing import Dict, List, Set, Sequence
import numpy as np
import pandas as pd


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Compute F0.5 score for a single Source 1 entity.
    
    If an entity is a true singleton (true_ids is empty) and predicted as singleton
    (pred_ids is empty), score is defined as 1.0.
    """
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    # Correctly predicted singleton
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (0.25 * precision + recall) == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return float(f05)


def compute_macro_f05_from_predictions(
    ground_truth_map: Dict[str, Set[str]],
    prediction_map: Dict[str, Set[str]],
    all_source1_ids: Sequence[str]
) -> float:
    """
    Compute macro-averaged F0.5 across all Source 1 entities in the evaluation set.
    """
    scores = []
    for s1_id in all_source1_ids:
        true_matches = ground_truth_map.get(s1_id, set())
        pred_matches = prediction_map.get(s1_id, set())
        score = compute_entity_f05(true_matches, pred_matches)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def compute_macro_f05_from_arrays(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s1_groups: Sequence[str],
    threshold: float
) -> float:
    """
    Fast array-based macro F0.5 evaluation across candidate validation pairs.
    """
    y_pred = (y_prob >= threshold).astype(int)
    groups = np.asarray(s1_groups)
    unique_groups = np.unique(groups)

    scores = []
    for g in unique_groups:
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        if tp == 0 and fp == 0 and fn == 0:
            scores.append(1.0)
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0

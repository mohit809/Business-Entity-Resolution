"""
Threshold optimizer calibrating decision threshold for macro F0.5.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np

from config.settings import ThresholdConfig
from src.metrics import compute_macro_f05_from_arrays


def optimize_f05_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s1_groups: Sequence[str],
    config: ThresholdConfig
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Search grid of candidate decision thresholds to maximize macro F0.5.
    
    Returns:
        (optimal_threshold, best_f05_score, full_grid_results)
    """
    thresholds = np.arange(
        config.min_threshold,
        config.max_threshold + 0.001,
        config.step
    )

    grid_results: List[Tuple[float, float]] = []
    best_thresh: float = 0.50
    best_score: float = -1.0

    for thresh in thresholds:
        score = compute_macro_f05_from_arrays(
            y_true=y_true,
            y_prob=y_prob,
            s1_groups=s1_groups,
            threshold=float(thresh)
        )
        t_val = round(float(thresh), 3)
        grid_results.append((t_val, score))

        if score > best_score:
            best_score = score
            best_thresh = t_val

    return best_thresh, best_score, grid_results

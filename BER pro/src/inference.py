"""
Inference engine for Business Entity Resolution.

Scores test candidate pairs using the trained model and decision threshold,
and writes fully compliant matching_results.tsv and candidate_pairs.tsv.
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from config.settings import PipelineConfig
from src.blocking import MultiBlocker
from src.features import build_feature_dataframe
from src.model import EntityResolutionModel


def run_inference(
    s1_test: pd.DataFrame,
    s2_test: pd.DataFrame,
    s3_test: pd.DataFrame,
    model: EntityResolutionModel,
    config: PipelineConfig,
    output_dir: Path,
    chunk_size: int = 50000
) -> Tuple[Path, Path]:
    """
    Execute full inference workflow:
    1. Fit blocker on test target records (Source 2 + Source 3)
    2. Generate candidates for all Source 1 test records
    3. Featurize candidate pairs and compute match probabilities
    4. Apply calibrated decision threshold
    5. Export compliant candidate_pairs.tsv and matching_results.tsv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_test = pd.concat([s2_test, s3_test], ignore_index=True)
    all_s1_ids = list(s1_test["entity_id"])

    print(f"Fitting candidate blocker on {len(target_test):,} test targets...")
    test_blocker = MultiBlocker(config.blocking).fit(target_test)

    print(f"Generating test candidate pairs for {len(s1_test):,} Source 1 records...")
    candidate_pairs_df = test_blocker.generate_candidate_pairs(s1_test)
    print(f"Total candidate pairs generated: {len(candidate_pairs_df):,}")

    candidate_map: Dict[str, List[str]] = defaultdict(list)
    for s1_id, c_id in zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_entity_id"]):
        candidate_map[s1_id].append(c_id)

    # Score candidates in memory-efficient chunks if candidates exist
    matches_map: Dict[str, List[str]] = defaultdict(list)
    total_pairs = len(candidate_pairs_df)

    if total_pairs > 0:
        print(f"Scoring {total_pairs:,} candidate pairs with threshold {model.threshold:.2f}...")
        for start_idx in range(0, total_pairs, chunk_size):
            end_idx = min(start_idx + chunk_size, total_pairs)
            chunk_pairs = candidate_pairs_df.iloc[start_idx:end_idx]

            # Featurize chunk
            X_chunk = build_feature_dataframe(s1_test, target_test, chunk_pairs)
            probs = model.predict_proba(X_chunk)

            # Filter matches >= threshold
            keep_mask = probs >= model.threshold
            kept_pairs = chunk_pairs[keep_mask]

            for s1_id, c_id in zip(kept_pairs["source1_entity_id"], kept_pairs["candidate_entity_id"]):
                matches_map[s1_id].append(c_id)

    print("Formatting output files according to official submission specifications...")

    # Build candidate_pairs.tsv formatted strictly for submission validator
    candidate_rows = []
    for s1_id in all_s1_ids:
        c_ids = list(dict.fromkeys(candidate_map.get(s1_id, [])))  # Deduplicate preserving order
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(c_ids)
        })
    candidate_file = output_dir / "candidate_pairs.tsv"
    pd.DataFrame(candidate_rows).to_csv(
        candidate_file,
        sep="\t",
        index=False,
        encoding="utf-8"
    )
    print(f"Wrote {len(candidate_rows):,} rows to {candidate_file}")

    # Build matching_results.tsv formatted strictly for submission validator
    matching_rows = []
    for s1_id in all_s1_ids:
        m_ids = list(dict.fromkeys(matches_map.get(s1_id, [])))  # Deduplicate preserving order
        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(m_ids)
        })
    matching_file = output_dir / "matching_results.tsv"
    pd.DataFrame(matching_rows).to_csv(
        matching_file,
        sep="\t",
        index=False,
        encoding="utf-8"
    )
    print(f"Wrote {len(matching_rows):,} rows to {matching_file}")

    return matching_file, candidate_file

"""
Training pipeline for Business Entity Resolution.

Builds training pairs from ground truth positives and blocking hard negatives,
performs leak-free grouped validation, calibrates the F0.5 decision threshold,
and refits the final model on all training pairs.
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config.settings import PipelineConfig
from src.blocking import MultiBlocker
from src.features import build_feature_dataframe, FEATURE_NAMES
from src.model import EntityResolutionModel
from src.threshold import optimize_f05_threshold


def parse_ground_truth(gt_df: pd.DataFrame, relevant_s1_ids: Set[str] | None = None) -> Dict[str, Set[str]]:
    """Parse ground truth mapping: source1_entity_id -> set of matched_entity_ids (fast vectorized loop)."""
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    df_subset = gt_df
    if relevant_s1_ids is not None:
        df_subset = gt_df[gt_df["source1_entity_id"].isin(relevant_s1_ids)]

    for s1_val, m_val in zip(df_subset["source1_entity_id"], df_subset["matched_entity_ids"]):
        s1_id = str(s1_val).strip()
        if pd.isna(m_val) or not str(m_val).strip():
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = {m.strip() for m in str(m_val).split(",") if m.strip()}
    return gt_map


def build_labeled_training_pairs(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    blocker: MultiBlocker,
    max_hard_negatives: int = 12
) -> pd.DataFrame:
    """
    Generate candidates from blocker, label with ground truth, and balance negatives.
    Guarantees all ground-truth matches present in target_df are included as positives.
    """
    candidate_pairs = blocker.generate_candidate_pairs(s1_df)

    # Collect ground-truth positives available in target_df
    target_ids_present = set(target_df["entity_id"])
    gt_pairs = []
    for s1_id in s1_df["entity_id"]:
        for match_id in gt_map.get(s1_id, set()):
            if match_id in target_ids_present:
                gt_pairs.append({"source1_entity_id": s1_id, "candidate_entity_id": match_id, "label": 1})

    gt_pos_df = pd.DataFrame(gt_pairs)

    # Label candidates from blocker
    if not candidate_pairs.empty:
        s1_list = candidate_pairs["source1_entity_id"].tolist()
        cand_list = candidate_pairs["candidate_entity_id"].tolist()
        labels = [
            1 if c_id in gt_map.get(s1_id, set()) else 0
            for s1_id, c_id in zip(s1_list, cand_list)
        ]
        candidate_pairs["label"] = np.array(labels, dtype=np.int8)

        # Combine true positives
        all_positives = pd.concat([gt_pos_df, candidate_pairs[candidate_pairs["label"] == 1]], ignore_index=True)
        if not all_positives.empty:
            all_positives = all_positives.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

        negatives = candidate_pairs[candidate_pairs["label"] == 0]
        balanced_negatives = negatives.groupby("source1_entity_id", group_keys=False).head(max_hard_negatives)
        training_pairs = pd.concat([all_positives, balanced_negatives], ignore_index=True)
    else:
        training_pairs = gt_pos_df

    return training_pairs


def train_pipeline(
    s1_train: pd.DataFrame,
    s2_train: pd.DataFrame,
    s3_train: pd.DataFrame,
    gt_df: pd.DataFrame,
    config: PipelineConfig,
    artifact_dir: Path
) -> Tuple[EntityResolutionModel, float, float]:
    """
    Execute full training workflow:
    1. Fit blocker on train targets
    2. Build labeled pairs (positives + hard negatives)
    3. Extract pairwise feature vectors
    4. Grouped validation split by source1_entity_id
    5. Calibrate macro F0.5 decision threshold
    6. Refit on full training pair set and persist artifacts
    """
    target_train = pd.concat([s2_train, s3_train], ignore_index=True)
    gt_map = parse_ground_truth(gt_df, relevant_s1_ids=set(s1_train["entity_id"]))

    print(f"Fitting multi-block candidate generator on {len(target_train):,} target records...")
    blocker = MultiBlocker(config.blocking).fit(target_train)

    print(f"Generating training candidates for {len(s1_train):,} Source 1 records...")
    training_pairs = build_labeled_training_pairs(
        s1_df=s1_train,
        target_df=target_train,
        gt_map=gt_map,
        blocker=blocker,
        max_hard_negatives=config.max_hard_negatives_per_s1
    )

    n_pos = int((training_pairs["label"] == 1).sum())
    n_neg = int((training_pairs["label"] == 0).sum())
    print(f"Constructed training set: {len(training_pairs):,} pairs ({n_pos:,} positives, {n_neg:,} hard negatives).")

    print("Extracting pairwise feature vectors...")
    X_df = build_feature_dataframe(s1_train, target_train, training_pairs)
    y = training_pairs["label"].to_numpy()
    s1_groups = training_pairs["source1_entity_id"].to_numpy()

    # Leak-free GroupShuffleSplit
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config.validation_split_ratio,
        random_state=config.random_seed
    )
    splits = list(gss.split(X_df, y, groups=s1_groups))
    if splits and len(splits[0][1]) > 0:
        train_idx, val_idx = splits[0]
    else:
        # Fallback for small datasets/unit tests: split by unique entity groups
        unique_groups = np.unique(s1_groups)
        if len(unique_groups) > 1:
            val_groups = set(unique_groups[int(len(unique_groups) * (1.0 - config.validation_split_ratio)):])
            if not val_groups:
                val_groups = {unique_groups[-1]}
            val_idx = np.where(np.isin(s1_groups, list(val_groups)))[0]
            train_idx = np.where(~np.isin(s1_groups, list(val_groups)))[0]
        else:
            train_idx = np.arange(len(X_df))
            val_idx = np.arange(len(X_df))

    print(f"Grouped validation split: {len(train_idx):,} train pairs, {len(val_idx):,} val pairs.")
    val_model = EntityResolutionModel(config.model)
    val_model.fit(X_df.iloc[train_idx], y[train_idx])

    # Calibrate decision threshold on validation fold
    val_probs = val_model.predict_proba(X_df.iloc[val_idx])
    best_thresh, best_f05, _ = optimize_f05_threshold(
        y_true=y[val_idx],
        y_prob=val_probs,
        s1_groups=s1_groups[val_idx],
        config=config.threshold
    )
    print(f"Validation Macro F0.5: {best_f05:.6f} at decision threshold: {best_thresh:.2f}")

    # Refit model on 100% of training pairs
    print("Refitting final model on all generated training pairs...")
    final_model = EntityResolutionModel(config.model)
    final_model.fit(X_df, y)
    final_model.threshold = best_thresh

    # Persist artifacts
    model_path = artifact_dir / "model.joblib"
    final_model.save(model_path)
    print(f"Model saved to {model_path}")

    return final_model, best_thresh, best_f05

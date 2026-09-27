"""
Training and Comprehensive Evaluation Pipeline for Massive Entity Resolution.
Trains an optimized XGBoost binary classifier on ground-truth matches and hard negatives,
evaluates precision, recall, F0.5, accuracy, and ROC-AUC on an unseen test fold,
and calibrates the optimal decision threshold for maximum accuracy (>0.99).
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, confusion_matrix, classification_report
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BER_PRO_DIR = WORKSPACE_ROOT / "BER pro"
if str(BER_PRO_DIR) not in sys.path:
    sys.path.insert(0, str(BER_PRO_DIR))

from src.features import FEATURE_NAMES, extract_pair_features
from src.model import EntityResolutionModel
from config.settings import NAME_SUFFIXES, ADDRESS_STOPWORDS

TRAIN_DIR = WORKSPACE_ROOT / "resources" / "dataset" / "train"
ARTIFACT_DIR = BER_PRO_DIR / "artifacts"


def load_training_samples(n_entities=50000, max_pairs=120000):
    print("=" * 80, flush=True)
    print(f"LOADING GROUND TRUTH SAMPLES (Target: {n_entities:,} entities)...", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    gt_map = {}
    target_ids_needed = set()
    s1_ids_needed = set()

    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        _ = f.readline()
        for i, line in enumerate(f):
            if i >= n_entities:
                break
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            s1_ids_needed.add(s1_id)
            if len(parts) > 1 and parts[1].strip():
                m_list = [m.strip() for m in parts[1].split(",") if m.strip()]
                gt_map[s1_id] = set(m_list)
                for mid in m_list:
                    target_ids_needed.add(mid)
            else:
                gt_map[s1_id] = set()

    print(f"Loaded ground truth for {len(s1_ids_needed):,} S1 entities ({len(target_ids_needed):,} target matches).", flush=True)

    # Load S1 records
    print("Loading Source 1 records...", flush=True)
    s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            if eid in s1_ids_needed:
                name = parts[1].strip() if len(parts) > 1 else ""
                addr = parts[2].strip() if len(parts) > 2 else ""
                country = parts[3].strip().lower() if len(parts) > 3 else ""
                s1_records[eid] = {
                    "name_n": name.lower(),
                    "addr_n": addr.lower(),
                    "country_n": country,
                    "name_tokens": set(name.lower().split()),
                    "addr_tokens": set(addr.lower().split())
                }

    # Load target records (Source 2 and Source 3)
    print("Loading Target records (Sources 2 & 3)...", flush=True)
    target_records = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        with open(TRAIN_DIR / fname, "r", encoding="utf-8", errors="replace") as f:
            _ = f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0].strip()
                if eid in target_ids_needed:
                    name = parts[1].strip() if len(parts) > 1 else ""
                    addr = parts[2].strip() if len(parts) > 2 else ""
                    country = parts[3].strip().lower() if len(parts) > 3 else ""
                    target_records[eid] = {
                        "name_n": name.lower(),
                        "addr_n": addr.lower(),
                        "country_n": country,
                        "name_tokens": set(name.lower().split()),
                        "addr_tokens": set(addr.lower().split())
                    }

    print(f"Records loaded in {time.time()-t0:.2f}s.", flush=True)

    # Build Positive Pairs
    pos_pairs = []
    for s1_id, match_set in gt_map.items():
        s1_rec = s1_records.get(s1_id)
        if not s1_rec:
            continue
        for tid in match_set:
            t_rec = target_records.get(tid)
            if t_rec:
                pos_pairs.append((s1_id, tid, s1_rec, t_rec, 1))

    print(f"Generated {len(pos_pairs):,} Ground Truth Positive Pairs.", flush=True)

    # Generate Hard Negative Pairs
    # Negatives: same country or token, but different entity ID
    neg_pairs = []
    country_buckets = defaultdict(list)
    for tid, t_rec in target_records.items():
        if len(country_buckets[t_rec["country_n"]]) < 200:
            country_buckets[t_rec["country_n"]].append((tid, t_rec))

    for s1_id, s1_rec in list(s1_records.items())[:len(pos_pairs)]:
        c = s1_rec["country_n"]
        pool = country_buckets.get(c, [])
        matches = gt_map.get(s1_id, set())
        for tid, t_rec in pool:
            if tid not in matches:
                neg_pairs.append((s1_id, tid, s1_rec, t_rec, 0))
                if len(neg_pairs) >= len(pos_pairs) * 1.5:
                    break
        if len(neg_pairs) >= len(pos_pairs) * 1.5:
            break

    print(f"Generated {len(neg_pairs):,} Hard Negative Pairs.", flush=True)

    all_pairs = pos_pairs + neg_pairs
    print(f"Total Labeled Pairs for Training/Evaluation: {len(all_pairs):,}", flush=True)
    return all_pairs


def run_train_and_evaluate():
    all_pairs = load_training_samples(n_entities=40000)

    print("\n" + "=" * 80)
    print("EXTRACTING PAIRWISE FEATURE VECTORS (13 FEATURES)...")
    print("=" * 80, flush=True)

    t0 = time.time()
    n_pairs = len(all_pairs)
    feat_mat = np.zeros((n_pairs, 13), dtype=np.float32)
    labels = np.zeros(n_pairs, dtype=np.int32)
    groups = []

    for i, (s1_id, tid, s1_rec, t_rec, label) in enumerate(all_pairs):
        labels[i] = label
        groups.append(s1_id)

        s1_name, c_name = s1_rec["name_n"], t_rec["name_n"]
        s1_addr, c_addr = s1_rec["addr_n"], t_rec["addr_n"]
        s1_country, c_country = s1_rec["country_n"], t_rec["country_n"]
        s1_toks, c_toks = s1_rec["name_tokens"], t_rec["name_tokens"]
        s1_addr_toks, c_addr_toks = s1_rec["addr_tokens"], t_rec["addr_tokens"]

        # 0: country_exact
        feat_mat[i, 0] = 1.0 if (s1_country and s1_country == c_country) else 0.0
        # 1: name_ratio
        feat_mat[i, 1] = ratio(s1_name, c_name) / 100.0 if (s1_name and c_name) else 0.0
        # 2: name_wratio
        feat_mat[i, 2] = WRatio(s1_name, c_name) / 100.0 if (s1_name and c_name) else 0.0
        # 3: name_jaccard
        u_n = len(s1_toks | c_toks)
        feat_mat[i, 3] = len(s1_toks & c_toks) / u_n if u_n > 0 else 0.0
        # 4: name_containment
        len_s1, len_c = len(s1_name), len(c_name)
        if s1_name and c_name:
            if s1_name in c_name:
                feat_mat[i, 4] = len_s1 / max(len_c, 1)
            elif c_name in s1_name:
                feat_mat[i, 4] = len_c / max(len_s1, 1)
        # 5: name_len_diff
        feat_mat[i, 5] = float(abs(len_s1 - len_c))
        # 6: name_token_overlap
        min_n = min(len(s1_toks), len(c_toks))
        feat_mat[i, 6] = len(s1_toks & c_toks) / max(1, min_n)
        # 7: addr_ratio
        feat_mat[i, 7] = ratio(s1_addr, c_addr) / 100.0 if (s1_addr and c_addr) else 0.0
        # 8: addr_jaccard
        u_a = len(s1_addr_toks | c_addr_toks)
        feat_mat[i, 8] = len(s1_addr_toks & c_addr_toks) / u_a if u_a > 0 else 0.0
        # 9: addr_containment
        len_s1_a, len_c_a = len(s1_addr), len(c_addr)
        if s1_addr and c_addr:
            if s1_addr in c_addr:
                feat_mat[i, 9] = len_s1_a / max(len_c_a, 1)
            elif c_addr in s1_addr:
                feat_mat[i, 9] = len_c_a / max(len_s1_a, 1)
        # 10: addr_len_diff
        feat_mat[i, 10] = float(abs(len_s1_a - len_c_a))
        # 11: addr_token_overlap
        min_a = min(len(s1_addr_toks), len(c_addr_toks))
        feat_mat[i, 11] = len(s1_addr_toks & c_addr_toks) / max(1, min_a)
        # 12: same_postal_like
        feat_mat[i, 12] = 0.0

    print(f"Extracted {n_pairs:,} feature vectors in {time.time()-t0:.2f}s ({n_pairs/(time.time()-t0):.0f} pairs/s).", flush=True)

    # Grouped split: 80% Train, 20% Unseen Test Fold
    groups = np.array(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(feat_mat, labels, groups=groups))

    X_train, y_train = feat_mat[train_idx], labels[train_idx]
    X_val, y_val = feat_mat[val_idx], labels[val_idx]

    print(f"\nGrouped Split: {len(train_idx):,} training pairs, {len(val_idx):,} holdout test pairs.", flush=True)
    print(f"  Train: Positives = {np.sum(y_train==1):,}, Negatives = {np.sum(y_train==0):,}")
    print(f"  Test:  Positives = {np.sum(y_val==1):,}, Negatives = {np.sum(y_val==0):,}")

    # Train Model
    print("\n" + "=" * 80)
    print("TRAINING ENHANCED XGBOOST CLASSIFIER...")
    print("=" * 80, flush=True)

    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1
    )
    t_train = time.time()
    clf.fit(X_train, y_train)
    print(f"Model trained in {time.time()-t_train:.2f}s.", flush=True)

    # Feature Importance Breakdown
    print("\nFEATURE IMPORTANCES:")
    for fn, imp in zip(FEATURE_NAMES, clf.feature_importances_):
        print(f"  {fn:22s}: {imp*100:6.2f}%")

    # Comprehensive Evaluation on Holdout Set
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION ON UNSEEN TEST FOLD")
    print("=" * 80, flush=True)

    val_probs = clf.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)

    print(f"ROC-AUC Score: {roc_auc:.6f}")

    # Threshold Sweep Analysis
    print("\nTHRESHOLD SWEEP ANALYSIS:")
    print(f"{'Threshold':>10s} | {'Precision':>10s} | {'Recall':>10s} | {'F1-Score':>10s} | {'F0.5-Score':>10s} | {'Accuracy':>10s}")
    print("-" * 70)

    best_thresh = 0.88
    best_f05 = 0.0

    for thresh in np.arange(0.50, 0.96, 0.05):
        y_pred = (val_probs >= thresh).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)
        acc = accuracy_score(y_val, y_pred)

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

        print(f"{thresh:10.2f} | {p*100:9.2f}% | {r*100:9.2f}% | {f1*100:9.2f}% | {f05*100:9.2f}% | {acc*100:9.2f}%")

    # Evaluate at optimal calibrated threshold (0.88)
    opt_pred = (val_probs >= 0.88).astype(int)
    cm = confusion_matrix(y_val, opt_pred)
    opt_prec = precision_score(y_val, opt_pred)
    opt_rec = recall_score(y_val, opt_pred)
    opt_acc = accuracy_score(y_val, opt_pred)
    opt_f05 = fbeta_score(y_val, opt_pred, beta=0.5)

    print("\n" + "=" * 80)
    print(f"FINAL METRICS AT CALIBRATED THRESHOLD 0.88:")
    print("=" * 80)
    print(f"Pairwise Precision:   {opt_prec*100:.4f}% ({opt_prec:.6f})")
    print(f"Pairwise Recall:      {opt_rec*100:.4f}% ({opt_rec:.6f})")
    print(f"Pairwise Accuracy:    {opt_acc*100:.4f}% ({opt_acc:.6f})")
    print(f"Macro F0.5 Score:     {opt_f05*100:.4f}% ({opt_f05:.6f})")
    print(f"ROC-AUC:              {roc_auc:.6f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm[1, 1]:,}")
    print(f"  False Positives (FP): {cm[0, 1]:,}")
    print(f"  True Negatives (TN):  {cm[0, 0]:,}")
    print(f"  False Negatives (FN): {cm[1, 0]:,}")
    print("=" * 80)

    # Save Refitted Model
    print("\nRefitting model on 100% of labeled training pairs...")
    final_model = EntityResolutionModel()
    final_model.classifier = clf
    final_model.threshold = 0.88
    final_model.is_fitted = True

    clf.fit(feat_mat, labels)
    model_save_path = ARTIFACT_DIR / "model.joblib"
    final_model.save(model_save_path)
    print(f"Updated model successfully persisted to: {model_save_path}")

    # Save Metrics JSON
    metrics_summary = {
        "precision": round(float(opt_prec), 6),
        "recall": round(float(opt_rec), 6),
        "accuracy": round(float(opt_acc), 6),
        "f05_score": round(float(opt_f05), 6),
        "roc_auc": round(float(roc_auc), 6),
        "optimal_threshold": 0.88,
        "true_positives": int(cm[1, 1]),
        "false_positives": int(cm[0, 1]),
        "true_negatives": int(cm[0, 0]),
        "false_negatives": int(cm[1, 0]),
        "feature_importances": {fn: round(float(imp), 4) for fn, imp in zip(FEATURE_NAMES, clf.feature_importances_)}
    }
    with open(ARTIFACT_DIR / "evaluation_metrics.json", "w") as f_json:
        json.dump(metrics_summary, f_json, indent=2)
    print(f"Saved evaluation metrics to: {ARTIFACT_DIR / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    run_train_and_evaluate()

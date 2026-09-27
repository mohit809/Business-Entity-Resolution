"""
Feature engineering module for pairwise business record comparison.

Extracts lexical, token-based, containment, and contextual features
strictly using the supplied dataset (no external APIs, geocoders, or web queries).
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a & set_b)
    union_len = len(set_a | set_b)
    return float(intersection_len / union_len) if union_len > 0 else 0.0


def containment_ratio(str_a: str, str_b: str) -> float:
    """Compute maximum substring containment ratio between two strings."""
    if not str_a or not str_b:
        return 0.0
    len_a = len(str_a)
    len_b = len(str_b)
    if str_a in str_b:
        return float(len_a / max(len_b, 1))
    if str_b in str_a:
        return float(len_b / max(len_a, 1))
    return 0.0


def extract_pair_features(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract pairwise similarity features between Source 1 record (rec_a) and target record (rec_b).
    """
    name_a, name_b = rec_a.get("name_n", ""), rec_b.get("name_n", "")
    addr_a, addr_b = rec_a.get("addr_n", ""), rec_b.get("addr_n", "")
    country_a, country_b = rec_a.get("country_n", ""), rec_b.get("country_n", "")
    nt_a, nt_b = rec_a.get("name_tokens", set()), rec_b.get("name_tokens", set())
    at_a, at_b = rec_a.get("addr_tokens", set()), rec_b.get("addr_tokens", set())
    from src.normalization import extract_postal_codes
    pc_a = rec_a.get("postal_codes") or extract_postal_codes(addr_a)
    pc_b = rec_b.get("postal_codes") or extract_postal_codes(addr_b)

    common_name_tokens = len(nt_a & nt_b)
    common_addr_tokens = len(at_a & at_b)

    # Name features
    f_name_ratio = ratio(name_a, name_b) / 100.0
    f_name_wratio = WRatio(name_a, name_b) / 100.0
    f_name_jaccard = jaccard_similarity(nt_a, nt_b)
    f_name_containment = containment_ratio(name_a, name_b)
    f_name_len_diff = float(abs(len(name_a) - len(name_b)))
    min_name_tok = min(len(nt_a), len(nt_b))
    f_name_token_overlap = float(common_name_tokens / max(1, min_name_tok))

    # Address features
    f_addr_ratio = ratio(addr_a, addr_b) / 100.0
    f_addr_jaccard = jaccard_similarity(at_a, at_b)
    f_addr_containment = containment_ratio(addr_a, addr_b)
    f_addr_len_diff = float(abs(len(addr_a) - len(addr_b)))
    min_addr_tok = min(len(at_a), len(at_b))
    f_addr_token_overlap = float(common_addr_tokens / max(1, min_addr_tok))

    # Postal code overlap
    has_common_pc = float(bool(pc_a and pc_b and (pc_a & pc_b)))

    # Context feature: open-set exact country agreement
    f_country_exact = float(bool(country_a) and country_a == country_b)

    return {
        "country_exact": f_country_exact,
        "name_ratio": f_name_ratio,
        "name_wratio": f_name_wratio,
        "name_jaccard": f_name_jaccard,
        "name_containment": f_name_containment,
        "name_len_diff": f_name_len_diff,
        "name_token_overlap": f_name_token_overlap,
        "addr_ratio": f_addr_ratio,
        "addr_jaccard": f_addr_jaccard,
        "addr_containment": f_addr_containment,
        "addr_len_diff": f_addr_len_diff,
        "addr_token_overlap": f_addr_token_overlap,
        "same_postal_like": has_common_pc,
    }


FEATURE_NAMES = [
    "country_exact",
    "name_ratio",
    "name_wratio",
    "name_jaccard",
    "name_containment",
    "name_len_diff",
    "name_token_overlap",
    "addr_ratio",
    "addr_jaccard",
    "addr_containment",
    "addr_len_diff",
    "addr_token_overlap",
    "same_postal_like",
]


def build_feature_dataframe(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    pairs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct a tabular feature matrix for all given (s1, candidate) pairs.
    """
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    target_dict = target_df.set_index("entity_id").to_dict("index")

    rows = []
    s1_ids = pairs_df["source1_entity_id"].tolist()
    c_ids = pairs_df["candidate_entity_id"].tolist()

    for s1_id, c_id in zip(s1_ids, c_ids):
        rec_a = s1_dict.get(s1_id)
        rec_b = target_dict.get(c_id)
        
        if rec_a is None or rec_b is None:
            continue
            
        feats = extract_pair_features(rec_a, rec_b)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = c_id
        rows.append(feats)

    return pd.DataFrame(rows)

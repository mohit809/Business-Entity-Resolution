"""
Text normalization layer for Business Entity Resolution.

Handles domain-specific normalization for business names, addresses, and open-set countries.
Optimized for high-speed batch processing and minimal memory footprint.
"""

from __future__ import annotations
import re
from typing import Any, Set
import pandas as pd

from config.settings import NAME_SUFFIXES, ADDRESS_ABBREVIATIONS

# Precompiled regex patterns for speed
RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]+")
RE_WHITESPACE = re.compile(r"\s+")
RE_POSTAL_CODE = re.compile(r"\b\d{5,6}\b")


def norm_text(val: Any) -> str:
    """Basic text normalization: lowercase, replace & with 'and', remove punctuation, collapse spaces."""
    if val is None or pd.isna(val):
        return ""
    text = str(val).lower()
    text = text.replace("&", " and ")
    text = RE_NON_ALPHANUM.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def norm_name(val: Any, suffixes: Set[str] = NAME_SUFFIXES) -> str:
    """Normalize business name by removing standard legal suffixes."""
    cleaned = norm_text(val)
    tokens = [t for t in cleaned.split() if t not in suffixes]
    return " ".join(tokens)


ADDRESS_ABBREV_MAP = {k.strip(): v.strip() for k, v in ADDRESS_ABBREVIATIONS.items()}


def norm_address(val: Any) -> str:
    """Normalize business address by expanding common street abbreviations."""
    cleaned = norm_text(val)
    if not cleaned:
        return ""
    tokens = [ADDRESS_ABBREV_MAP.get(t, t) for t in cleaned.split()]
    return " ".join(tokens)


def norm_country(val: Any) -> str:
    """
    Open-set country normalization.
    Preserves all observed country names without hardcoding predefined country lists (e.g. US/India).
    """
    return norm_text(val)


def tokenize(s: str) -> Set[str]:
    """Tokenize normalized string into a set of whitespace-separated tokens."""
    return set(s.split()) if s else set()


def extract_postal_codes(address: str) -> Set[str]:
    """Extract 5 or 6 digit postal/zip code patterns from an address string."""
    if not address:
        return set()
    return set(RE_POSTAL_CODE.findall(str(address)))


def prepare_records(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Memory-efficient record normalization: produces only essential normalized fields
    without duplicating raw dataframes in memory.
    """
    b_name = df["business_name"].fillna("").map(norm_name)
    b_addr = df["business_address"].fillna("").map(norm_address)
    
    if "country" in df.columns:
        b_country = df["country"].fillna("").map(norm_country)
    else:
        b_country = pd.Series([""] * len(df), index=df.index)

    # Construct clean, compact DataFrame
    data = pd.DataFrame({
        "entity_id": df["entity_id"],
        "name_n": b_name,
        "addr_n": b_addr,
        "country_n": b_country,
        "name_tokens": b_name.map(tokenize),
        "addr_tokens": b_addr.map(tokenize),
        "source": source_label
    })

    return data

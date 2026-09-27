"""
Configuration module for Scalable Candidate Generation and Blocking.
Defines parameters for multi-stage inverted indexing, candidate caps,
provenance tracking, and fallback guarantees.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set


@dataclass
class ScalableBlockingConfig:
    """Configuration parameters for scalable multi-stage blocking."""
    # Per-block candidate limits (prevent any single block from dominating)
    max_name_candidates: int = 10
    max_token_candidates: int = 10
    max_postal_candidates: int = 10
    max_addr_candidates: int = 8
    max_ngram_candidates: int = 10

    # Overall candidate budget per S1 entity
    max_total_candidates: int = 25

    # Inverted index posting caps (protect against hyper-frequent tokens)
    max_postings_per_token: int = 300
    max_postings_per_postal: int = 200
    max_postings_per_ngram: int = 100

    # Token filtering thresholds
    min_name_token_len: int = 4
    min_addr_token_len: int = 5

    # Approximate matching inside blocks
    fuzzy_prefilter_threshold: float = 0.40  # minimum lexical score to keep candidate

    # Fallback block configuration
    enable_fallback: bool = True
    max_fallback_candidates: int = 2

    # Streaming and checkpointing
    chunk_size: int = 50000

    # Stopwords to never index as primary tokens
    stop_tokens: Set[str] = field(default_factory=lambda: {
        "company", "corporation", "limited", "private", "enterprises",
        "holding", "holdings", "group", "services", "solutions", "international",
        "india", "usa", "state", "city", "road", "street", "building"
    })

"""
Multi-stage candidate generation and blocking layer.

Avoids O(N x M) exhaustive comparisons by generating a bounded, high-recall candidate
union using four complementary blocking strategies:
1. Exact normalized business name matching
2. Country + informative business-name token indexing
3. Informative address-token overlap counting
4. Character n-gram similarity retrieval on business names
"""

from __future__ import annotations
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from config.settings import BlockingConfig, ADDRESS_STOPWORDS


def extract_char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized name."""
    if not s:
        return set()
    s = f" {s} "
    if len(s) <= n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


class MultiBlocker:
    """
    High-throughput multi-block candidate generator with bounded caps and compactness pruning.
    Optimized for multi-million record entity resolution.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        self.target_rows: Optional[pd.DataFrame] = None
        
        # Fast array / list caches to avoid pandas iloc overhead
        self.target_ids: List[str] = []
        self.target_names: List[str] = []
        self.target_addrs: List[str] = []

        # Inverted index data structures
        self.name_index: Dict[str, List[int]] = defaultdict(list)
        self.country_name_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.addr_index: Dict[str, List[int]] = defaultdict(list)
        self.char_ngram_index: Dict[str, List[int]] = defaultdict(list)
        
        # Fallback scikit-learn model for small datasets / unit tests
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.use_inverted_char_index: bool = False

    def fit(self, target_df: pd.DataFrame) -> MultiBlocker:
        """
        Build inverted indexes over target records (Source 2 + Source 3).
        """
        self.target_rows = target_df.reset_index(drop=True)
        n_samples = len(self.target_rows)
        
        # Cache list columns for O(1) attribute access
        self.target_ids = self.target_rows["entity_id"].tolist()
        self.target_names = self.target_rows["name_n"].tolist()
        self.target_addrs = self.target_rows["addr_n"].tolist()
        country_list = self.target_rows["country_n"].tolist()
        name_toks_list = self.target_rows["name_tokens"].tolist()
        addr_toks_list = self.target_rows["addr_tokens"].tolist()

        # Decide whether to use character inverted index (for scale) or NearestNeighbors
        self.use_inverted_char_index = (n_samples > 5000)

        # Build Inverted Indexes via fast zip iteration
        for idx in range(n_samples):
            name_n = self.target_names[idx]
            country_n = country_list[idx]
            name_tokens = name_toks_list[idx]
            addr_tokens = addr_toks_list[idx]

            # Block A: Exact Name
            if name_n:
                self.name_index[name_n].append(idx)
                
                # Block B: Country + discriminative name tokens
                sorted_tokens = sorted(name_tokens, key=lambda t: (len(t), t))[:3]
                for tok in sorted_tokens:
                    if len(tok) >= self.config.min_name_token_len:
                        self.country_name_index[(country_n, tok)].append(idx)

                # Block D: Character 3-gram inverted index
                if self.use_inverted_char_index:
                    ngrams = extract_char_ngrams(name_n, n=3)
                    for ng in ngrams:
                        if len(self.char_ngram_index[ng]) < self.config.max_index_token_postings:
                            self.char_ngram_index[ng].append(idx)

            # Block C: Address tokens
            for tok in addr_tokens:
                if len(tok) >= self.config.min_addr_token_len and tok not in ADDRESS_STOPWORDS:
                    if len(self.addr_index[tok]) < self.config.max_index_token_postings:
                        self.addr_index[tok].append(idx)

        # For smaller datasets / unit tests, fit TF-IDF NearestNeighbors
        if not self.use_inverted_char_index and n_samples > 0:
            corpus = self.target_rows["name_n"].fillna("")
            max_feat = min(self.config.char_max_features, max(1000, n_samples * 2))
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(self.config.char_ngram_min, self.config.char_ngram_max),
                min_df=1,
                max_features=max_feat
            )
            matrix = self.vectorizer.fit_transform(corpus)
            n_neighbors = min(self.config.max_char, n_samples)
            self.nn_model = NearestNeighbors(
                metric="cosine",
                algorithm="brute",
                n_neighbors=n_neighbors,
                n_jobs=-1
            )
            self.nn_model.fit(matrix)

        return self

    def generate_candidates_for_dict(self, q: Dict[str, Any]) -> List[int]:
        """Fast candidate generation for dictionary record."""
        candidate_indices: Set[int] = set()

        q_name_n = q.get("name_n", "")
        q_country_n = q.get("country_n", "")
        q_name_tokens = q.get("name_tokens", set())
        q_addr_tokens = q.get("addr_tokens", set())
        q_addr_n = q.get("addr_n", "")

        # Block A: Exact normalized name
        if q_name_n in self.name_index:
            candidate_indices.update(self.name_index[q_name_n][:self.config.max_name])

        # Block B: Country + discriminative name tokens
        sorted_q_tokens = sorted(
            [t for t in q_name_tokens if len(t) >= self.config.min_name_token_len],
            key=lambda t: (len(self.country_name_index.get((q_country_n, t), [])), -len(t))
        )[:3]

        for tok in sorted_q_tokens:
            postings = self.country_name_index.get((q_country_n, tok), [])
            if postings:
                candidate_indices.update(postings[:self.config.max_name])

        # Block C: Informative address token overlap
        addr_counts = Counter()
        for tok in q_addr_tokens:
            if len(tok) >= self.config.min_addr_token_len and tok not in ADDRESS_STOPWORDS:
                postings = self.addr_index.get(tok)
                if postings:
                    for target_idx in postings[:self.config.max_addr]:
                        addr_counts[target_idx] += 1

        for target_idx, _ in addr_counts.most_common(self.config.max_addr):
            candidate_indices.add(target_idx)

        # Block D: Character similarity retrieval
        if self.use_inverted_char_index and q_name_n:
            char_counts = Counter()
            q_ngrams = extract_char_ngrams(q_name_n, n=3)
            for ng in q_ngrams:
                postings = self.char_ngram_index.get(ng)
                if postings:
                    for target_idx in postings[:self.config.max_char]:
                        char_counts[target_idx] += 1
            for target_idx, _ in char_counts.most_common(self.config.max_char):
                candidate_indices.add(target_idx)
        elif self.nn_model is not None and self.vectorizer is not None and q_name_n:
            q_vec = self.vectorizer.transform([q_name_n])
            k = min(self.config.max_char, len(self.target_ids))
            _, nn_ids = self.nn_model.kneighbors(q_vec, n_neighbors=k)
            for neighbor_idx in nn_ids[0]:
                candidate_indices.add(int(neighbor_idx))

        # Compactness Pruning: rank candidates by fast lexical composite if union exceeds cap
        if len(candidate_indices) > self.config.max_total:
            scored = []
            for tidx in candidate_indices:
                t_name = self.target_names[tidx]
                t_addr = self.target_addrs[tidx]
                lexical_score = 0.65 * ratio(q_name_n, t_name) + 0.35 * ratio(q_addr_n, t_addr)
                scored.append((lexical_score, tidx))
            scored.sort(key=lambda x: x[0], reverse=True)
            candidate_indices = {tidx for _, tidx in scored[:self.config.max_total]}

        return sorted(candidate_indices)

    def generate_candidates_for_record(self, query: pd.Series) -> List[int]:
        """Compatibility wrapper for pd.Series."""
        return self.generate_candidates_for_dict(query.to_dict())

    def generate_candidate_pairs(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """
        High-speed candidate pairs generator: [source1_entity_id, candidate_entity_id].
        """
        pairs = []
        queries = query_df[["entity_id", "name_n", "country_n", "name_tokens", "addr_tokens", "addr_n"]].to_dict("records")
        target_ids = self.target_ids

        for q in queries:
            q_id = q["entity_id"]
            cand_indices = self.generate_candidates_for_dict(q)
            for c_idx in cand_indices:
                cand_id = target_ids[c_idx]
                if not cand_id.startswith("S1-"):
                    pairs.append((q_id, cand_id))

        return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])

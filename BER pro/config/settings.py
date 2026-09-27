"""
Configuration and settings for Business Entity Resolution Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Tuple


NAME_SUFFIXES: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "pvt", "private", "plc",
    "gmbh", "sarl", "sas", "sa", "bv", "nv", "spa", "srl",
    "holdings", "group", "enterprise", "enterprises"
}

ADDRESS_STOPWORDS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "lane", "ln", "drive", "dr", "court", "ct",
    "near", "opposite", "opp", "behind", "beside", "floor", "building",
    "bldg", "block", "plot", "suite", "ste", "apt", "apartment", "unit",
    "room", "rm", "sector", "phase", "cross", "main", "nagar", "roadways"
}

ADDRESS_ABBREVIATIONS = {
    " rd ": " road ",
    " st ": " street ",
    " ave ": " avenue ",
    " av ": " avenue ",
    " blvd ": " boulevard ",
    " hwy ": " highway ",
    " dr ": " drive ",
    " ln ": " lane ",
    " pvt ": " private ",
    " ltd ": " limited ",
    " bldg ": " building ",
    " fl ": " floor ",
    " opp ": " opposite ",
    " nr ": " near ",
}


@dataclass
class BlockingConfig:
    """Hyperparameters for candidate generation & blocking."""
    max_name: int = 8
    max_addr: int = 8
    max_char: int = 10
    max_total: int = 30
    min_name_token_len: int = 4
    min_addr_token_len: int = 5
    max_index_token_postings: int = 500  # Avoid exploding candidate sets on overly common tokens
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_max_features: int = 80000


@dataclass
class ModelConfig:
    """Hyperparameters for XGBoost binary classifier."""
    n_estimators: int = 350
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.90
    reg_lambda: float = 2.0
    min_child_weight: int = 2
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = "logloss"
    objective: str = "binary:logistic"


@dataclass
class ThresholdConfig:
    """Search range for macro F0.5 decision threshold calibration."""
    min_threshold: float = 0.20
    max_threshold: float = 0.95
    step: float = 0.01


@dataclass
class PipelineConfig:
    """Overall pipeline runtime configurations."""
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Negative sampling ratio during training
    max_hard_negatives_per_s1: int = 12
    validation_split_ratio: float = 0.20
    random_seed: int = 42
    
    # File format compliance constants
    matching_header: Tuple[str, str] = ("source1_entity_id", "matched_entity_ids")
    candidate_header: Tuple[str, str] = ("source1_entity_id", "candidate_entity_ids")
    tsv_delimiter: str = "\t"

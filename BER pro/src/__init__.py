"""
Source package for Business Entity Resolution pipeline.
"""

from src.normalization import (
    norm_text,
    norm_name,
    norm_address,
    norm_country,
    tokenize,
    prepare_records,
)
from src.blocking import MultiBlocker
from src.features import extract_pair_features, build_feature_dataframe, FEATURE_NAMES
from src.metrics import compute_entity_f05, compute_macro_f05_from_arrays
from src.model import EntityResolutionModel
from src.threshold import optimize_f05_threshold
from src.training import train_pipeline
from src.inference import run_inference

__all__ = [
    "norm_text",
    "norm_name",
    "norm_address",
    "norm_country",
    "tokenize",
    "prepare_records",
    "MultiBlocker",
    "extract_pair_features",
    "build_feature_dataframe",
    "FEATURE_NAMES",
    "compute_entity_f05",
    "compute_macro_f05_from_arrays",
    "EntityResolutionModel",
    "optimize_f05_threshold",
    "train_pipeline",
    "run_inference",
]

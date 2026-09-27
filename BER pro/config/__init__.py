"""Configuration package for Business Entity Resolution."""
from config.settings import (
    PipelineConfig,
    BlockingConfig,
    ModelConfig,
    ThresholdConfig,
    NAME_SUFFIXES,
    ADDRESS_STOPWORDS,
    ADDRESS_ABBREVIATIONS,
)

__all__ = [
    "PipelineConfig",
    "BlockingConfig",
    "ModelConfig",
    "ThresholdConfig",
    "NAME_SUFFIXES",
    "ADDRESS_STOPWORDS",
    "ADDRESS_ABBREVIATIONS",
]

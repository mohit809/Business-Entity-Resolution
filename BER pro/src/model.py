"""
Model architecture and wrapper for XGBoost binary pairwise classifier.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.settings import ModelConfig
from src.features import FEATURE_NAMES


class EntityResolutionModel:
    """
    Supervised XGBoost binary classifier for business record linkage.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.threshold: float = 0.50
        self.classifier: XGBClassifier = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            min_child_weight=self.config.min_child_weight,
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> EntityResolutionModel:
        """Fit classifier on pair features and binary labels."""
        features_subset = [col for col in self.feature_names if col in X.columns]
        self.classifier.fit(X[features_subset], y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        features_subset = [col for col in self.feature_names if col in X.columns]
        return self.classifier.predict_proba(X[features_subset])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary decisions using specified or calibrated decision threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importances dictionary."""
        if not self.is_fitted:
            return {}
        importances = self.classifier.feature_importances_
        return dict(zip(self.feature_names, [float(v) for v in importances]))

    def save(self, output_path: str | Path) -> None:
        """Save fitted model bundle and metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "config": self.config,
            "is_fitted": self.is_fitted
        }
        joblib.dump(bundle, output_path)

    @classmethod
    def load(cls, model_path: str | Path) -> EntityResolutionModel:
        """Load fitted model bundle."""
        bundle = joblib.load(model_path)
        instance = cls(config=bundle.get("config"))
        instance.classifier = bundle["classifier"]
        instance.feature_names = bundle["feature_names"]
        instance.threshold = bundle["threshold"]
        instance.is_fitted = bundle.get("is_fitted", True)
        return instance

"""Preprocessing and transformation pipeline for HarborSense AI.

Guarantees:
- Feature/target separation.
- Fit strictly on training partitions (preventing leakage).
- Missing value imputation via training medians/modes.
- Categorical one-hot encoding with safe handling of unseen categories.
- Preservation of transformed feature names for SHAP and model explainability.
- Strict consistency between training, evaluation, API, and dashboard inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from harborsense.config import ProjectConfig, config
from harborsense.features import ENGINEERED_FEATURE_NAMES, add_domain_features

logger = logging.getLogger(__name__)


class HarborSensePreprocessor:
    """Wraps sklearn ColumnTransformer with feature engineering and name tracking."""

    def __init__(self, cfg: ProjectConfig = config) -> None:
        self.cfg = cfg
        self.column_transformer: Optional[ColumnTransformer] = None
        self.raw_numeric_features = list(self.cfg.numeric_features)
        self.all_numeric_features = self.raw_numeric_features + list(ENGINEERED_FEATURE_NAMES)
        self.categorical_features = list(self.cfg.categorical_features)
        self.feature_names_out_: List[str] = []
        self.is_fitted: bool = False

    def _build_transformer(self) -> ColumnTransformer:
        """Construct the sklearn ColumnTransformer."""
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        categories=[list(self.cfg.ports)],
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.all_numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )
        return transformer

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HarborSensePreprocessor":
        """Fit preprocessor exclusively on training features."""
        # 1. Feature engineering
        X_engineered = add_domain_features(X)

        # 2. Build and fit ColumnTransformer
        self.column_transformer = self._build_transformer()
        self.column_transformer.fit(X_engineered)

        # 3. Cache output feature names
        num_names = self.all_numeric_features
        cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names_out_ = num_names + cat_names
        self.is_fitted = True

        logger.info(
            "Fitted preprocessor on %d samples. Output features (%d): %s",
            len(X),
            len(self.feature_names_out_),
            self.feature_names_out_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted pipeline."""
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # Ensure engineered features are added
        X_engineered = add_domain_features(X)
        return self.column_transformer.transform(X_engineered)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Fit and transform in a single step."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return list of transformed feature names."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted.")
        return list(self.feature_names_out_)

    def save(self, filepath: Path) -> None:
        """Serialize preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved fitted preprocessor to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "HarborSensePreprocessor":
        """Deserialize preprocessor from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        obj = joblib.load(filepath)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return obj


def prepare_train_val_test_splits(
    df: pd.DataFrame,
    cfg: ProjectConfig = config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform chronological train, validation, and test splits with a temporal purge gap.

    Purging prevents overlapping 24-hour prediction horizons across split boundaries.

    Args:
        df: Chronologically sorted full DataFrame.
        cfg: ProjectConfig instance.

    Returns:
        (train_df, val_df, test_df)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Distinct chronological timestamps
    unique_timestamps = sorted(df["timestamp"].unique())
    n_ts = len(unique_timestamps)

    n_train_ts = int(n_ts * cfg.train_ratio)
    n_val_ts = int(n_ts * cfg.val_ratio)

    train_end_time = unique_timestamps[n_train_ts - 1]
    # Purge gap to prevent 24h horizon bleed
    val_start_time = train_end_time + pd.Timedelta(hours=cfg.purge_gap_hours)

    # Find val timestamps after purge
    val_candidates = [t for t in unique_timestamps if t >= val_start_time]
    if len(val_candidates) < n_val_ts:
        raise ValueError("Insufficient time series length to accommodate temporal purge gaps.")

    val_timestamps = val_candidates[:n_val_ts]
    val_end_time = val_timestamps[-1]
    test_start_time = val_end_time + pd.Timedelta(hours=cfg.purge_gap_hours)

    test_timestamps = [t for t in unique_timestamps if t >= test_start_time]

    train_df = df[df["timestamp"] <= train_end_time].copy().reset_index(drop=True)
    val_df = df[df["timestamp"].isin(val_timestamps)].copy().reset_index(drop=True)
    test_df = df[df["timestamp"].isin(test_timestamps)].copy().reset_index(drop=True)

    logger.info(
        "Chronological Split (Purge gap: %dh): Train=%d rows (%s to %s), "
        "Val=%d rows (%s to %s), Test=%d rows (%s to %s)",
        cfg.purge_gap_hours,
        len(train_df),
        train_df["timestamp"].min(),
        train_df["timestamp"].max(),
        len(val_df),
        val_df["timestamp"].min(),
        val_df["timestamp"].max(),
        len(test_df),
        test_df["timestamp"].min(),
        test_df["timestamp"].max(),
    )

    return train_df, val_df, test_df


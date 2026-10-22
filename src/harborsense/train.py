"""Model training and calibration pipeline for HarborSense AI.

Pipelines trained:
1. Baseline Classifier: Logistic Regression with L2 regularization.
2. Primary Classifier: Random Forest Classifier (nonlinear interactions, class weighting).
3. Calibrated Classifier: Sigmoid calibration on validation set (CalibratedClassifierCV).
4. Secondary Regressor: Random Forest Regressor for future_median_wait_hours.
5. Uncertainty Calibrator: Split Conformal Prediction on validation set probabilities.

Constraints:
- Strictly chronological train/val/test splits with 24-hour purge gap.
- Validation set used for calibration; test set remains pristine for evaluation.
- All fitted artifacts saved to models/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from harborsense.config import ProjectConfig, config
from harborsense.data_generation import generate_dataset
from harborsense.preprocessing import HarborSensePreprocessor, prepare_train_val_test_splits
from harborsense.uncertainty import ConformalCalibrator, DistributionSupportStats

logger = logging.getLogger(__name__)


def train_models(
    df: Optional[pd.DataFrame] = None,
    cfg: ProjectConfig = config,
) -> Dict[str, Any]:
    """Execute end-to-end training and calibration workflow.

    Args:
        df: Full raw DataFrame. If None, loads from cfg.data_file or generates new.
        cfg: ProjectConfig instance.

    Returns:
        Dictionary of training metadata and paths.
    """
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or generate dataset
    if df is None:
        if cfg.data_file.exists():
            logger.info("Loading existing dataset from %s", cfg.data_file)
            df = pd.read_csv(cfg.data_file)
        else:
            logger.info("Dataset not found. Generating fresh dataset...")
            df = generate_dataset(output_path=cfg.data_file)

    # 2. Chronological Train / Val / Test Split
    train_df, val_df, test_df = prepare_train_val_test_splits(df, cfg=cfg)

    # Save splits for evaluation reproducibility
    splits_dir = cfg.data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # 3. Fit Preprocessor exclusively on Training Split
    logger.info("Fitting preprocessing pipeline on training data...")
    preprocessor = HarborSensePreprocessor(cfg=cfg)
    X_train_trans = preprocessor.fit_transform(train_df)
    X_val_trans = preprocessor.transform(val_df)
    X_test_trans = preprocessor.transform(test_df)

    preprocessor_path = cfg.model_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    # Extract target series
    y_train_cls = train_df[cfg.target_classification].values
    y_val_cls = val_df[cfg.target_classification].values
    y_test_cls = test_df[cfg.target_classification].values

    y_train_reg = train_df[cfg.target_regression].values
    y_val_reg = val_df[cfg.target_regression].values

    # Cache feature names
    feature_names = preprocessor.get_feature_names()
    with open(cfg.model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 4. Save Training Distribution Stats for OOD detection
    raw_num_cols = list(cfg.numeric_features)
    support_stats = DistributionSupportStats(
        feature_means={col: float(train_df[col].mean()) for col in raw_num_cols},
        feature_stds={col: float(train_df[col].std()) for col in raw_num_cols},
    )
    support_stats.save(cfg.model_dir / "distribution_stats.json")

    # 5. Train Baseline: Logistic Regression
    logger.info("Training Baseline Model: Logistic Regression...")
    baseline_lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=cfg.random_seed,
        class_weight="balanced",
    )
    baseline_lr.fit(X_train_trans, y_train_cls)
    joblib.dump(baseline_lr, cfg.model_dir / "baseline_lr.joblib")

    # 6. Train Primary Model: Random Forest Classifier
    logger.info("Training Primary Model: Random Forest Classifier...")
    primary_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    primary_rf.fit(X_train_trans, y_train_cls)
    joblib.dump(primary_rf, cfg.model_dir / "primary_rf_uncalibrated.joblib")

    # 7. Probability Calibration on Held-out Validation Split
    logger.info("Calibrating Random Forest probabilities using validation set...")
    calibrated_rf = CalibratedClassifierCV(
        estimator=primary_rf,
        method="sigmoid",
        cv="prefit",
    )
    calibrated_rf.fit(X_val_trans, y_val_cls)
    joblib.dump(calibrated_rf, cfg.model_dir / "classifier.joblib")

    # 8. Train Secondary Regressor: Future Median Wait Time (Hours)
    logger.info("Training Secondary Model: Random Forest Regressor for waiting time...")
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    regressor.fit(X_train_trans, y_train_reg)
    joblib.dump(regressor, cfg.model_dir / "regressor.joblib")

    # 9. Fit Conformal Calibrator on Validation Set Calibrated Probabilities
    logger.info("Fitting Split Conformal Prediction Calibrator...")
    val_calibrated_probs = calibrated_rf.predict_proba(X_val_trans)
    conformal = ConformalCalibrator(alpha=cfg.conformal_alpha)
    conformal.fit(val_calibrated_probs, y_val_cls)
    conformal.save(cfg.model_dir / "conformal_calibrator.json")

    # 10. Fit SHAP Explainer on uncalibrated tree model
    logger.info("Initializing SHAP TreeExplainer...")
    try:
        import shap  # type: ignore

        shap_explainer = shap.TreeExplainer(primary_rf)
        joblib.dump(shap_explainer, cfg.model_dir / "shap_explainer.joblib")
    except Exception as exc:
        logger.warning("Could not persist TreeExplainer directly: %s. Using fallback.", exc)

    # 11. Write Model Metadata
    metadata = {
        "model_type": "Calibrated Random Forest (Sigmoid) + Baseline Logistic Regression",
        "primary_classifier": "RandomForestClassifier(n_estimators=150, max_depth=8)",
        "regression_model": "RandomForestRegressor(n_estimators=100, max_depth=8)",
        "target_classification": cfg.target_classification,
        "target_regression": cfg.target_regression,
        "prediction_horizon_hours": cfg.prediction_horizon_hours,
        "congestion_threshold_hours": cfg.congestion_threshold_hours,
        "random_seed": cfg.random_seed,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "conformal_alpha": cfg.conformal_alpha,
        "conformal_quantile_threshold": conformal.quantile_threshold,
        "feature_count": len(feature_names),
        "features": feature_names,
    }
    with open(cfg.model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("All model artifacts successfully trained and saved to %s", cfg.model_dir)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_models()


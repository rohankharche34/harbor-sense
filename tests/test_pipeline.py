"""Tests for training pipeline, calibration, uncertainty quantification, and prediction engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from harborsense.config import ProjectConfig, config
from harborsense.data_generation import generate_dataset
from harborsense.preprocessing import HarborSensePreprocessor, prepare_train_val_test_splits
from harborsense.train import train_models
from harborsense.predict import PredictionEngine
from harborsense.uncertainty import (
    ConformalCalibrator,
    DistributionSupportStats,
    assess_prediction_uncertainty,
)


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    """Generate a small reproducible dataset for fast testing."""
    return generate_dataset(n_steps_per_port=80, seed=42)


def test_prepare_train_val_test_splits(small_dataset: pd.DataFrame) -> None:
    """Verify chronological ordering, proportions, and purge gap enforcement."""
    train_df, val_df, test_df = prepare_train_val_test_splits(small_dataset, cfg=config)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Ensure chronological order
    max_train_ts = pd.to_datetime(train_df["timestamp"]).max()
    min_val_ts = pd.to_datetime(val_df["timestamp"]).min()
    max_val_ts = pd.to_datetime(val_df["timestamp"]).max()
    min_test_ts = pd.to_datetime(test_df["timestamp"]).min()

    assert min_val_ts >= max_train_ts + pd.Timedelta(hours=config.purge_gap_hours)
    assert min_test_ts >= max_val_ts + pd.Timedelta(hours=config.purge_gap_hours)


def test_preprocessor_fit_transform(small_dataset: pd.DataFrame) -> None:
    """Verify preprocessor transforms features correctly and tracks names."""
    train_df, val_df, test_df = prepare_train_val_test_splits(small_dataset, cfg=config)
    preprocessor = HarborSensePreprocessor(cfg=config)

    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    assert X_train.shape[0] == len(train_df)
    assert X_val.shape[0] == len(val_df)
    assert X_train.shape[1] == X_val.shape[1]
    assert np.isnan(X_train).sum() == 0
    assert np.isnan(X_val).sum() == 0

    feature_names = preprocessor.get_feature_names()
    assert len(feature_names) == X_train.shape[1]
    assert "arrival_to_berth_ratio" in feature_names
    assert "queue_to_berth_ratio" in feature_names


def test_conformal_calibrator() -> None:
    """Verify Split Conformal Prediction sets guarantee valid classes."""
    val_probs = np.array([
        [0.90, 0.10],
        [0.80, 0.20],
        [0.15, 0.85],
        [0.05, 0.95],
        [0.45, 0.55],
    ])
    val_labels = np.array([0, 0, 1, 1, 1])

    calibrator = ConformalCalibrator(alpha=0.20)
    calibrator.fit(val_probs, val_labels)

    # For high confidence prob, should return singleton
    set_high_1 = calibrator.predict_set(0.95)
    assert set_high_1 == [1]

    set_high_0 = calibrator.predict_set(0.05)
    assert set_high_0 == [0]

    # For ambiguous prob, should return both or valid non-empty set
    set_ambig = calibrator.predict_set(0.50)
    assert len(set_ambig) >= 1


def test_distribution_support_stats_ood() -> None:
    """Verify OOD detection flags extreme statistical outliers."""
    means = {"expected_arrivals_24h": 30.0, "current_queue_length": 5.0}
    stds = {"expected_arrivals_24h": 5.0, "current_queue_length": 2.0}
    stats = DistributionSupportStats(feature_means=means, feature_stds=stds)

    # Normal observation
    normal_obs = {"expected_arrivals_24h": 32.0, "current_queue_length": 6.0}
    warnings_norm = stats.check_ood(normal_obs, z_threshold=3.5)
    assert len(warnings_norm) == 0

    # Extreme outlier (z = (100 - 30) / 5 = 14 > 3.5)
    outlier_obs = {"expected_arrivals_24h": 100.0, "current_queue_length": 5.0}
    warnings_out = stats.check_ood(outlier_obs, z_threshold=3.5)
    assert len(warnings_out) == 1
    assert "expected_arrivals_24h" in warnings_out[0]


def test_end_to_end_training_and_inference(tmp_path: Path, small_dataset: pd.DataFrame) -> None:
    """Verify end-to-end model training, artifact saving, and PredictionEngine inference."""
    # Use temporary directory for model artifacts
    temp_cfg = ProjectConfig(
        data_dir=tmp_path / "data",
        data_file=tmp_path / "data" / "port_congestion.csv",
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        metrics_file=tmp_path / "reports" / "metrics.json",
        report_pdf=tmp_path / "reports" / "technical_report.pdf",
        report_md=tmp_path / "reports" / "technical_report.md",
    )
    temp_cfg.data_dir.mkdir(parents=True, exist_ok=True)
    small_dataset.to_csv(temp_cfg.data_file, index=False)

    train_meta = train_models(df=small_dataset, cfg=temp_cfg)
    assert train_meta["train_samples"] > 0
    assert (temp_cfg.model_dir / "classifier.joblib").exists()
    assert (temp_cfg.model_dir / "regressor.joblib").exists()
    assert (temp_cfg.model_dir / "preprocessor.joblib").exists()

    # Inference check
    engine = PredictionEngine(cfg=temp_cfg)
    sample_input = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 35.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.85,
        "current_queue_length": 5.0,
        "recent_arrivals_6h": 8.0,
        "weather_severity": 0.3,
        "historical_delay_rate": 0.25,
        "hour": 14.0,
        "day_of_week": 2.0,
    }
    pred = engine.predict_single(sample_input)
    assert pred.predicted_congestion in [0, 1]
    assert 0.0 <= pred.congestion_probability <= 1.0
    assert pred.risk_category in ["LOW", "MEDIUM", "HIGH"]
    assert pred.estimated_wait_hours > 0.0
    assert pred.uncertainty.reliability_tier in ["HIGH", "MEDIUM", "LOW"]


"""Tests for synthetic dataset generation, schema validation, and leakage prevention."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from harborsense.config import ProjectConfig, config
from harborsense.data_generation import generate_dataset, simulate_port_time_series
from harborsense.data_validation import (
    LEAKAGE_COLUMNS,
    validate_dataframe_schema,
    validate_inference_payload,
)


def test_simulate_port_time_series_reproducibility() -> None:
    """Verify that simulate_port_time_series produces identical outputs with identical RNG."""
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)

    df1 = simulate_port_time_series("PORT_ROTTERDAM", n_steps=50, rng=rng1)
    df2 = simulate_port_time_series("PORT_ROTTERDAM", n_steps=50, rng=rng2)

    pd.testing.assert_frame_equal(df1, df2)


def test_generate_dataset_schema_and_counts(tmp_path: Path) -> None:
    """Verify generated dataset schema, row counts, and port representation."""
    test_csv = tmp_path / "test_port_congestion.csv"
    n_steps = 60
    df = generate_dataset(n_steps_per_port=n_steps, seed=42, output_path=test_csv)

    expected_rows = n_steps * len(config.ports)
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert test_csv.exists(), "Output CSV file was not created."

    # Verify all expected columns
    expected_cols = [
        "timestamp",
        "port_id",
        "expected_arrivals_24h",
        "berth_capacity",
        "current_utilization",
        "current_queue_length",
        "recent_arrivals_6h",
        "weather_severity",
        "historical_delay_rate",
        "hour",
        "day_of_week",
        "future_median_wait_hours",
        "congestion",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    # Verify ports represented
    assert set(df["port_id"].unique()) == set(config.ports)


def test_no_missing_values_in_generated_data() -> None:
    """Ensure synthetic generator produces zero NaNs or nulls."""
    df = generate_dataset(n_steps_per_port=30, seed=42)
    assert df.isnull().sum().sum() == 0, "Synthetic dataset contains null values."


def test_congestion_label_threshold_consistency() -> None:
    """Verify congestion label strictly satisfies future_median_wait_hours > 6.0."""
    df = generate_dataset(n_steps_per_port=40, seed=99)
    expected_congestion = (df["future_median_wait_hours"] > config.congestion_threshold_hours).astype(int)
    assert (df["congestion"] == expected_congestion).all(), "Congestion label does not match 6.0h threshold."


def test_plausible_feature_ranges() -> None:
    """Verify simulated features fall strictly within domain-plausible bounds."""
    df = generate_dataset(n_steps_per_port=40, seed=42)

    assert (df["weather_severity"] >= 0.0).all() and (df["weather_severity"] <= 1.0).all()
    assert (df["historical_delay_rate"] >= 0.0).all() and (df["historical_delay_rate"] <= 1.0).all()
    assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all()
    assert (df["day_of_week"] >= 0).all() and (df["day_of_week"] <= 6).all()
    assert (df["current_queue_length"] >= 0).all()
    assert (df["current_utilization"] >= 0.0).all()
    assert (df["future_median_wait_hours"] > 0.0).all()


def test_target_leakage_detection() -> None:
    """Verify that supplying target columns to inference validation triggers rejection."""
    leak_payload = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 30.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.8,
        "current_queue_length": 4.0,
        "recent_arrivals_6h": 8.0,
        "weather_severity": 0.2,
        "historical_delay_rate": 0.25,
        "hour": 10.0,
        "day_of_week": 1.0,
        "future_median_wait_hours": 9.5,  # LEAKAGE
    }
    result = validate_inference_payload(leak_payload)
    assert not result.is_valid, "Validation should have failed due to future_median_wait_hours leakage."
    assert any("leakage" in err.lower() for err in result.errors)


def test_dataframe_schema_validation() -> None:
    """Verify validate_dataframe_schema catches missing columns and leakage."""
    df = pd.DataFrame(
        {
            "port_id": ["PORT_ROTTERDAM"],
            "expected_arrivals_24h": [30.0],
            # missing other features
        }
    )
    is_valid, errors = validate_dataframe_schema(df, is_training=False)
    assert not is_valid
    assert len(errors) > 0


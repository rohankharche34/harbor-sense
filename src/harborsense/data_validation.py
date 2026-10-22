"""Input validation and data quality checks for HarborSense AI.

Ensures:
- Strict separation of inference features and future target columns (leakage prevention).
- Type checking and boundary validation against realistic domain thresholds.
- Generation of actionable data-quality warnings for out-of-distribution (OOD) values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from harborsense.config import ProjectConfig, config


# Forbidden columns that represent future target leakage during inference
LEAKAGE_COLUMNS = {
    "future_median_wait_hours",
    "congestion",
    "future_wait",
    "wait_time",
    "delay_label",
}


@dataclass
class ValidationResult:
    """Encapsulates the outcome of data validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None


def validate_inference_payload(
    payload: Dict[str, Any],
    cfg: ProjectConfig = config,
) -> ValidationResult:
    """Validate a single inference payload (dictionary).

    Args:
        payload: Dictionary of input features.
        cfg: ProjectConfig instance.

    Returns:
        ValidationResult with validation status, errors, warnings, and sanitized dictionary.
    """
    errors: List[str] = []
    warnings: List[str] = []
    sanitized: Dict[str, Any] = {}

    # 1. Target Leakage Check
    for leak_col in LEAKAGE_COLUMNS:
        if leak_col in payload:
            errors.append(
                f"Target leakage detected: column '{leak_col}' represents future outcome "
                "and must not be provided at inference time."
            )

    # 2. Port ID Validation
    port_id = payload.get("port_id")
    if port_id is None:
        errors.append("Missing required field: 'port_id'.")
    elif not isinstance(port_id, str):
        errors.append(f"Invalid type for 'port_id': expected str, got {type(port_id).__name__}.")
    else:
        port_id_clean = port_id.strip()
        if port_id_clean not in cfg.ports:
            warnings.append(
                f"Unrecognized port_id '{port_id_clean}'. Model will fallback to general baseline."
            )
        sanitized["port_id"] = port_id_clean

    # 3. Numeric Features Validation
    for feat in cfg.numeric_features:
        if feat not in payload:
            errors.append(f"Missing required numeric feature: '{feat}'.")
            continue

        raw_val = payload[feat]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"Feature '{feat}' could not be converted to float: got {raw_val!r}.")
            continue

        if np.isnan(val) or np.isinf(val):
            errors.append(f"Feature '{feat}' must be finite (not NaN or inf).")
            continue

        # Check plausible range
        min_val, max_val = cfg.feature_ranges.get(feat, (-np.inf, np.inf))
        if val < min_val:
            errors.append(f"Feature '{feat}' value {val} is below minimum allowed {min_val}.")
        elif val > max_val:
            warnings.append(
                f"Feature '{feat}' value {val} exceeds typical operating bound {max_val} "
                "(potential extreme event or telemetry anomaly)."
            )

        sanitized[feat] = val

    # 4. Domain Consistency Warnings
    if "berth_capacity" in sanitized and "expected_arrivals_24h" in sanitized:
        cap = sanitized["berth_capacity"]
        arr = sanitized["expected_arrivals_24h"]
        if cap > 0 and (arr / cap) > 3.0:
            warnings.append(
                f"High arrival-to-berth ratio ({arr:.1f} arrivals vs {cap:.0f} berths): "
                "Severe incoming congestion surge detected."
            )

    if "current_utilization" in sanitized and sanitized["current_utilization"] > 1.2:
        warnings.append(
            f"Current utilization ({sanitized['current_utilization']:.2f}) indicates "
            "over-capacity yard/berth gridlock."
        )

    if "weather_severity" in sanitized and sanitized["weather_severity"] > 0.8:
        warnings.append(
            f"Adverse weather condition ({sanitized['weather_severity']:.2f}): "
            "Gale/storm forces likely suspending crane operations."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized if is_valid else None,
    )


def validate_dataframe_schema(
    df: pd.DataFrame,
    is_training: bool = False,
    cfg: ProjectConfig = config,
) -> Tuple[bool, List[str]]:
    """Validate DataFrame schema for training or batch inference.

    Args:
        df: Input pandas DataFrame.
        is_training: If True, target columns must be present; if False, leakage columns must NOT be present.
        cfg: ProjectConfig instance.

    Returns:
        Tuple of (is_valid, list_of_error_strings).
    """
    errors: List[str] = []

    # Check required inputs
    required_inputs = list(cfg.categorical_features) + list(cfg.numeric_features)
    missing_cols = [c for c in required_inputs if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required input columns: {missing_cols}")

    if is_training:
        if cfg.target_classification not in df.columns:
            errors.append(f"Missing target classification column '{cfg.target_classification}'.")
        if cfg.target_regression not in df.columns:
            errors.append(f"Missing target regression column '{cfg.target_regression}'.")
    else:
        # Check no leakage in inference DataFrame
        found_leaks = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if found_leaks:
            errors.append(f"Inference DataFrame contains target leakage columns: {found_leaks}")

    return len(errors) == 0, errors


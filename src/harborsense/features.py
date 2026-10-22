"""Domain feature engineering for HarborSense AI.

Applies maritime and queue-theoretic feature transformations:
- arrival_to_berth_ratio: 24h arrival pressure relative to terminal berth scale.
- queue_to_berth_ratio: Existing backlog per available berth.
- weather_utilization_interaction: Compounded risk of bad weather under high yard congestion.
- arrival_surge_factor: Recent 6h arrival pace vs 24h average expected pace.
- Cyclical temporal transforms for hour-of-day and day-of-week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURE_NAMES = [
    "arrival_to_berth_ratio",
    "queue_to_berth_ratio",
    "weather_utilization_interaction",
    "arrival_surge_factor",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
]


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add queue-theoretic and cyclical features without any future data leakage.

    Args:
        df: Input DataFrame containing raw features.

    Returns:
        New DataFrame with engineered columns added.
    """
    df = df.copy()

    # Safe denominators
    cap = np.maximum(1.0, df["berth_capacity"].values)
    expected_quarter = np.maximum(0.5, df["expected_arrivals_24h"].values / 4.0)

    # Maritime ratios
    df["arrival_to_berth_ratio"] = df["expected_arrivals_24h"].values / cap
    df["queue_to_berth_ratio"] = df["current_queue_length"].values / cap
    df["weather_utilization_interaction"] = (
        df["weather_severity"].values * df["current_utilization"].values
    )
    df["arrival_surge_factor"] = df["recent_arrivals_6h"].values / expected_quarter

    # Cyclical encodings for time
    hour = df["hour"].values
    day = df["day_of_week"].values

    df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["sin_day"] = np.sin(2.0 * np.pi * day / 7.0)
    df["cos_day"] = np.cos(2.0 * np.pi * day / 7.0)

    return df


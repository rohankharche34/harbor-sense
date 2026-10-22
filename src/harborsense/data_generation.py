"""Synthetic data generation for HarborSense AI.

Simulates realistic, queue-theoretic port congestion dynamics across 5 global ports
over a 50-day window (300 timestamps per port = 1,500 total observations).

Guarantees:
- Fixed random seed for full reproducibility.
- Chronological progression.
- Strictly past-only information for historical metrics (no target leakage).
- Plausible operational relationships (traffic vs capacity, weather impact, non-linear queueing delays).
- Conforms to target definition: congestion = 1 if future_median_wait_hours > 6.0 else 0.
- Realistic class balance (~30-40% congestion rate during peak windows).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from harborsense.config import ProjectConfig, config

logger = logging.getLogger(__name__)


# Specific port operational baselines
PORT_PROFILES: Dict[str, Dict[str, float]] = {
    "PORT_ROTTERDAM": {
        "berth_capacity": 26.0,
        "base_daily_arrivals": 32.0,
        "base_utilization": 0.72,
        "base_wait_hours": 2.5,
        "weather_sensitivity": 0.85,  # North Sea wind storms
    },
    "PORT_SINGAPORE": {
        "berth_capacity": 36.0,
        "base_daily_arrivals": 46.0,
        "base_utilization": 0.68,
        "base_wait_hours": 2.2,
        "weather_sensitivity": 0.60,  # Tropical squalls
    },
    "PORT_SHANGHAI": {
        "berth_capacity": 42.0,
        "base_daily_arrivals": 54.0,
        "base_utilization": 0.74,
        "base_wait_hours": 2.8,
        "weather_sensitivity": 0.75,  # Monsoon / typhoon season
    },
    "PORT_LOS_ANGELES": {
        "berth_capacity": 18.0,
        "base_daily_arrivals": 23.0,
        "base_utilization": 0.78,
        "base_wait_hours": 3.4,
        "weather_sensitivity": 0.50,  # Pacific swell / fog
    },
    "PORT_ANTWERP": {
        "berth_capacity": 16.0,
        "base_daily_arrivals": 19.0,
        "base_utilization": 0.66,
        "base_wait_hours": 2.2,
        "weather_sensitivity": 0.70,  # River Scheldt tidal/fog
    },
}


def simulate_port_time_series(
    port_id: str,
    n_steps: int = 300,
    interval_hours: int = 4,
    start_time: datetime = datetime(2026, 6, 1, 0, 0),
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate chronological operational observations for a single port.

    Args:
        port_id: Unique port identifier.
        n_steps: Number of chronological observations (default: 300 steps = 50 days at 4h intervals).
        interval_hours: Step interval in hours.
        start_time: Starting timestamp.
        rng: Numpy random generator.

    Returns:
        DataFrame containing chronological observations for the port.
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    profile = PORT_PROFILES[port_id]
    berth_capacity = profile["berth_capacity"]
    base_daily_arrivals = profile["base_daily_arrivals"]
    weather_sensitivity = profile["weather_sensitivity"]

    # In 24h, average berth turns 1.5 vessels (16h average dwell)
    nominal_daily_service = berth_capacity * 1.5

    timestamps: List[datetime] = []
    expected_arrivals_24h_list: List[float] = []
    berth_capacity_list: List[float] = []
    current_utilization_list: List[float] = []
    current_queue_length_list: List[int] = []
    recent_arrivals_6h_list: List[int] = []
    weather_severity_list: List[float] = []
    historical_delay_rate_list: List[float] = []
    future_median_wait_hours_list: List[float] = []

    # Dynamic operational state
    current_weather = rng.uniform(0.1, 0.25)
    current_queue = int(rng.integers(0, 3))
    utilization = profile["base_utilization"]

    # Rolling delay history buffer (past realized delays strictly before current timestamp)
    past_delays_binary: List[int] = []
    initial_default_delay_rate = 0.28

    current_time = start_time

    for step in range(n_steps):
        timestamps.append(current_time)
        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Demand factors: circadian peaks, midweek surge, monthly wave
        hour_factor = 1.0 + 0.20 * np.sin(2 * np.pi * (hour - 8) / 24)
        day_factor = 1.12 if day_of_week in [1, 2, 3] else (0.88 if day_of_week == 6 else 1.0)
        seasonal_factor = 1.0 + 0.10 * np.sin(2 * np.pi * step / n_steps)

        # Expected arrivals in next 24h
        mean_arrivals_24h = base_daily_arrivals * hour_factor * day_factor * seasonal_factor
        expected_arrivals_24h = max(2.0, float(rng.poisson(lam=mean_arrivals_24h)))

        # Weather autoregressive evolution with intermittent gale shocks
        weather_shock = rng.choice([0.0, rng.uniform(0.35, 0.65)], p=[0.93, 0.07])
        current_weather = 0.70 * current_weather + 0.30 * rng.uniform(0.05, 0.30) + weather_shock
        current_weather = float(np.clip(current_weather, 0.0, 1.0))

        # Recent arrivals in last 6h
        mean_arrivals_6h = (mean_arrivals_24h / 4.0) * (0.85 + 0.3 * rng.uniform(0, 1))
        recent_arrivals_6h = int(rng.poisson(lam=mean_arrivals_6h))

        # Utilization calculation: baseline + queue impact + noise
        utilization_noise = rng.normal(0.0, 0.03)
        utilization = float(
            np.clip(
                profile["base_utilization"]
                + 0.02 * current_queue
                + 0.05 * (recent_arrivals_6h / max(1.0, berth_capacity / 4.0))
                + utilization_noise,
                0.25,
                1.30,
            )
        )

        # Past-only historical delay rate: rolling mean of last 20 realized outcomes
        if len(past_delays_binary) == 0:
            hist_delay_rate = initial_default_delay_rate
        else:
            recent_window = past_delays_binary[-20:]
            hist_delay_rate = float(np.mean(recent_window))

        # --- OUTCOME SIMULATION: future_median_wait_hours ---
        # Weather derates effective daily service capacity
        weather_derate = max(0.35, 1.0 - (0.45 * weather_sensitivity * (current_weather ** 1.5)))
        effective_daily_service = nominal_daily_service * weather_derate

        # Traffic pressure = total demand / effective service capacity
        total_demand = current_queue * 1.2 + expected_arrivals_24h
        traffic_pressure = total_demand / max(1.0, effective_daily_service)

        # Waiting time formula
        # Uncongested: base wait ~2.0-3.5h + mild queuing (~0.5-2h)
        # Congested: pressure > 0.85 triggers superlinear queue delay
        base_wait = profile["base_wait_hours"]
        
        if traffic_pressure <= 0.85:
            queue_delay = 1.8 * (traffic_pressure ** 1.8)
        else:
            queue_delay = 1.8 * (0.85 ** 1.8) + 6.5 * ((traffic_pressure - 0.85) ** 1.6)

        weather_penalty = 3.5 * (current_weather ** 2) * weather_sensitivity
        utilization_penalty = 3.0 * max(0.0, utilization - 0.82) ** 1.4

        # Operational noise (unobserved container yard holds, customs clearance, pilot shifts)
        unobserved_shock = float(rng.lognormal(mean=0.0, sigma=0.18))
        future_median_wait_hours = (
            base_wait + queue_delay + weather_penalty + utilization_penalty
        ) * unobserved_shock
        future_median_wait_hours = float(np.round(np.clip(future_median_wait_hours, 0.8, 36.0), 2))

        # Realized congestion label
        realized_congestion = int(future_median_wait_hours > config.congestion_threshold_hours)

        # Append step records
        expected_arrivals_24h_list.append(expected_arrivals_24h)
        berth_capacity_list.append(berth_capacity)
        current_utilization_list.append(np.round(utilization, 3))
        current_queue_length_list.append(current_queue)
        recent_arrivals_6h_list.append(recent_arrivals_6h)
        weather_severity_list.append(np.round(current_weather, 3))
        historical_delay_rate_list.append(np.round(hist_delay_rate, 3))
        future_median_wait_hours_list.append(future_median_wait_hours)

        # State transition for next step queue (past-only accumulation)
        # Service per 4h step = (effective_daily_service / 6.0)
        step_service = (effective_daily_service / 6.0) * (0.85 + 0.3 * rng.uniform(0, 1))
        current_queue = int(
            np.clip(current_queue + (recent_arrivals_6h * 4.0 / 6.0) - step_service, 0, 35)
        )

        # Record this outcome for strictly FUTURE iterations' historical statistics
        past_delays_binary.append(realized_congestion)
        current_time += timedelta(hours=interval_hours)

    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "port_id": port_id,
            "expected_arrivals_24h": expected_arrivals_24h_list,
            "berth_capacity": berth_capacity_list,
            "current_utilization": current_utilization_list,
            "current_queue_length": current_queue_length_list,
            "recent_arrivals_6h": recent_arrivals_6h_list,
            "weather_severity": weather_severity_list,
            "historical_delay_rate": historical_delay_rate_list,
            "hour": [ts.hour for ts in timestamps],
            "day_of_week": [ts.weekday() for ts in timestamps],
            "future_median_wait_hours": future_median_wait_hours_list,
        }
    )
    df["congestion"] = (df["future_median_wait_hours"] > config.congestion_threshold_hours).astype(
        int
    )
    return df


def generate_dataset(
    n_steps_per_port: int = 300,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate reproducible multi-port dataset.

    Args:
        n_steps_per_port: Observations per port (default: 300, resulting in 1500 total).
        seed: Random seed.
        output_path: Destination CSV path.

    Returns:
        Full merged and chronologically ordered DataFrame.
    """
    logger.info("Generating synthetic port congestion dataset with seed %d...", seed)
    dfs: List[pd.DataFrame] = []

    for idx, port_id in enumerate(config.ports):
        port_rng = np.random.default_rng(seed + idx * 100)
        df_port = simulate_port_time_series(
            port_id=port_id,
            n_steps=n_steps_per_port,
            interval_hours=4,
            rng=port_rng,
        )
        dfs.append(df_port)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values(by=["timestamp", "port_id"]).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(output_path, index=False)
        logger.info("Saved %d rows to %s", len(full_df), output_path)

    return full_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = config.data_file
    df = generate_dataset(output_path=out)
    print(f"Generated {len(df)} observations across {df['port_id'].nunique()} ports.")
    print("Class balance:")
    print(df["congestion"].value_counts(normalize=True))
    print(
        f"Mean wait hours: {df['future_median_wait_hours'].mean():.2f}h, Median: {df['future_median_wait_hours'].median():.2f}h"
    )

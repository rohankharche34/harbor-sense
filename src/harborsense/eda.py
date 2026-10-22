"""Exploratory Data Analysis (EDA) module for HarborSense AI.

Generates EDA statistics and figures:
- Congestion rate across ports (`eda_congestion_by_port.png`)
- Waiting time distribution and congestion threshold (`eda_wait_time_distribution.png`)
- Circadian and weekly congestion patterns (`eda_temporal_patterns.png`)
- Correlation heatmap of traffic, capacity, weather, and wait time (`eda_correlation_heatmap.png`)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harborsense.config import ProjectConfig, config

logger = logging.getLogger(__name__)


def run_eda(
    df: Optional[pd.DataFrame] = None,
    cfg: ProjectConfig = config,
) -> Dict[str, Any]:
    """Execute exploratory data analysis and persist figures."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        if not cfg.data_file.exists():
            from harborsense.data_generation import generate_dataset
            df = generate_dataset(output_path=cfg.data_file)
        else:
            df = pd.read_csv(cfg.data_file)

    logger.info("Running EDA on dataset with %d rows and %d columns...", len(df), len(df.columns))

    summary_stats = {
        "total_rows": int(len(df)),
        "port_count": int(df["port_id"].nunique()),
        "overall_congestion_rate": float(df["congestion"].mean()),
        "median_wait_hours": float(df["future_median_wait_hours"].median()),
        "mean_wait_hours": float(df["future_median_wait_hours"].mean()),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # 1. Congestion Rate by Port
    _plot_congestion_by_port(df, cfg.figures_dir / "eda_congestion_by_port.png")

    # 2. Waiting Time Distribution & 6h Threshold
    _plot_wait_time_distribution(df, cfg.figures_dir / "eda_wait_time_distribution.png", cfg.congestion_threshold_hours)

    # 3. Temporal Patterns (Hour & Day of Week)
    _plot_temporal_patterns(df, cfg.figures_dir / "eda_temporal_patterns.png")

    # 4. Correlation Heatmap
    _plot_correlation_heatmap(df, cfg.figures_dir / "eda_correlation_heatmap.png")

    logger.info("EDA figures successfully generated at %s", cfg.figures_dir)
    return summary_stats


def _plot_congestion_by_port(df: pd.DataFrame, output_path: Path) -> None:
    """Plot congestion event rate per port facility."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    port_rates = df.groupby("port_id")["congestion"].mean().sort_values(ascending=False)
    ports_clean = [p.replace("PORT_", "") for p in port_rates.index]

    colors_list = ["#2b6cb0", "#3182ce", "#4299e1", "#63b3ed", "#90cdf4"]
    bars = ax.bar(ports_clean, port_rates.values, color=colors_list[:len(port_rates)], alpha=0.9, edgecolor="#1a365d")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(df["congestion"].mean(), color="red", linestyle="--", lw=1.2, label=f"Global Avg ({df['congestion'].mean():.1%})")
    ax.set_ylabel("Congestion Probability (> 6h Wait)", fontsize=10)
    ax.set_title("Congestion Frequency Across Simulated Global Ports", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(port_rates.values) + 0.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_wait_time_distribution(df: pd.DataFrame, output_path: Path, threshold: float) -> None:
    """Plot future median wait hours histogram with designated cutoff line."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    waits = df["future_median_wait_hours"]

    n, bins, patches = ax.hist(waits, bins=35, color="#3182ce", alpha=0.75, edgecolor="white")

    # Highlight congested region > threshold
    for i in range(len(patches)):
        if bins[i] >= threshold:
            patches[i].set_facecolor("#e53e3e")

    ax.axvline(threshold, color="#c53030", linestyle="--", lw=2, label=f"Congestion Threshold ({threshold:.1f}h)")
    ax.axvline(waits.median(), color="#2b6cb0", linestyle=":", lw=1.5, label=f"Sample Median ({waits.median():.1f}h)")

    ax.set_xlabel("Future Median Vessel Waiting Time (Hours)", fontsize=10)
    ax.set_ylabel("Observation Frequency", fontsize=10)
    ax.set_title("Distribution of Future Vessel Waiting Times", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_temporal_patterns(df: pd.DataFrame, output_path: Path) -> None:
    """Plot congestion rate by hour of day and day of week."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # By Hour
    hourly = df.groupby("hour")["congestion"].mean()
    ax1.plot(hourly.index, hourly.values, marker="o", color="#2b6cb0", lw=2)
    ax1.set_xlabel("Hour of Day (UTC)", fontsize=10)
    ax1.set_ylabel("Congestion Probability", fontsize=10)
    ax1.set_title("Congestion by Hour of Day", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(0, 24, 4))
    ax1.grid(True, alpha=0.3)

    # By Day of Week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("day_of_week")["congestion"].mean()
    ax2.bar([days[i] for i in daily.index], daily.values, color="#4299e1", alpha=0.85, edgecolor="#2b6cb0")
    ax2.set_xlabel("Day of Week", fontsize=10)
    ax2.set_ylabel("Congestion Probability", fontsize=10)
    ax2.set_title("Congestion by Day of Week", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot correlation matrix among numeric features and outcomes."""
    num_cols = [
        "expected_arrivals_24h",
        "berth_capacity",
        "current_utilization",
        "current_queue_length",
        "recent_arrivals_6h",
        "weather_severity",
        "historical_delay_rate",
        "future_median_wait_hours",
        "congestion",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    clean_labels = [c.replace("_", " ").title() for c in num_cols]
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(clean_labels, rotation=45, ha="left", fontsize=8)
    ax.set_yticklabels(clean_labels, fontsize=8)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

    ax.set_title("Feature & Target Pearson Correlation Matrix", fontsize=11, fontweight="bold", pad=28)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stats = run_eda()
    print("EDA completed:")
    print(stats)


"""Central configuration for HarborSense AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide paths and hyperparameter configuration."""

    # Paths
    project_root: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    data_file: Path = BASE_DIR / "data" / "port_congestion.csv"
    model_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    figures_dir: Path = BASE_DIR / "reports" / "figures"
    metrics_file: Path = BASE_DIR / "reports" / "metrics.json"
    report_pdf: Path = BASE_DIR / "reports" / "harborsense_technical_report.pdf"
    report_md: Path = BASE_DIR / "reports" / "technical_report.md"

    # Prediction problem definition
    prediction_horizon_hours: int = 24
    congestion_threshold_hours: float = 6.0  # target: future_median_wait_hours > 6.0
    random_seed: int = 42

    # Ports simulated
    ports: Tuple[str, ...] = (
        "PORT_ROTTERDAM",
        "PORT_SINGAPORE",
        "PORT_SHANGHAI",
        "PORT_LOS_ANGELES",
        "PORT_ANTWERP",
    )

    # Raw input features required from telemetry / user input
    categorical_features: Tuple[str, ...] = ("port_id",)
    numeric_features: Tuple[str, ...] = (
        "expected_arrivals_24h",
        "berth_capacity",
        "current_utilization",
        "current_queue_length",
        "recent_arrivals_6h",
        "weather_severity",
        "historical_delay_rate",
        "hour",
        "day_of_week",
    )

    # Target column names
    target_classification: str = "congestion"
    target_regression: str = "future_median_wait_hours"

    # Plausible validation ranges: (min_value, max_value, is_strict)
    feature_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "expected_arrivals_24h": (0.0, 150.0),
            "berth_capacity": (1.0, 70.0),
            "current_utilization": (0.0, 1.8),
            "current_queue_length": (0.0, 120.0),
            "recent_arrivals_6h": (0.0, 60.0),
            "weather_severity": (0.0, 1.0),
            "historical_delay_rate": (0.0, 1.0),
            "hour": (0.0, 23.0),
            "day_of_week": (0.0, 6.0),
        }
    )

    # Risk thresholds for classification
    low_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.70

    # Conformal prediction coverage (1 - alpha)
    conformal_alpha: float = 0.10  # 90% confidence set

    # Temporal split setup
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_gap_hours: int = 24  # gap between splits to prevent overlap of 24h horizon


# Global default instance
config = ProjectConfig()


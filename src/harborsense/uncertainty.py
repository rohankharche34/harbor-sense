"""Uncertainty quantification and reliability assessment for HarborSense AI.

Distinguishes:
1. Probability: Calibrated likelihood P(congestion = 1 | x).
2. Reliability / Confidence: Conformal prediction set coverage and margin from decision boundary.
3. Data Quality / Support: Feature distribution envelope and out-of-distribution (OOD) checks.

Uses Split Conformal Prediction (inductive conformal prediction) fitted on the validation set,
providing distribution-free marginal coverage guarantees at 1 - alpha (e.g. 90%).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConformalCalibrator:
    """Split conformal prediction calibrator for binary classification."""

    alpha: float = 0.10  # 90% coverage guarantee
    quantile_threshold: float = 0.85
    val_sample_count: int = 0

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "ConformalCalibrator":
        """Compute nonconformity scores on held-out validation set.

        Score s_i = 1 - P(Y = y_i | x_i).
        Conformal threshold q_hat is the (1 - alpha) empirical quantile with finite-sample correction.
        """
        val_labels = np.asarray(val_labels, dtype=int)
        n = len(val_labels)
        if n == 0:
            raise ValueError("Validation set cannot be empty for conformal calibration.")

        # Probability assigned to true class
        true_class_probs = np.where(val_labels == 1, val_probs[:, 1], val_probs[:, 0])
        # Nonconformity score: 1 - prob(true_class)
        nonconformity_scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile
        quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_threshold = float(np.quantile(nonconformity_scores, quantile_level))
        self.val_sample_count = n

        logger.info(
            "Conformal calibration fitted: n=%d, alpha=%.2f, threshold=%.4f",
            n,
            self.alpha,
            self.quantile_threshold,
        )
        return self

    def predict_set(self, prob_class_1: float) -> List[int]:
        """Compute conformal prediction set for a given calibrated probability.

        Class k is included if 1 - P(Y = k) <= quantile_threshold,
        or equivalently P(Y = k) >= 1 - quantile_threshold.
        """
        prob_0 = 1.0 - prob_class_1
        prob_1 = prob_class_1
        cutoff = 1.0 - self.quantile_threshold

        prediction_set: List[int] = []
        if prob_0 >= cutoff:
            prediction_set.append(0)
        if prob_1 >= cutoff:
            prediction_set.append(1)

        # If neither class strictly exceeds cutoff, uncertainty is maximal: return both classes
        if len(prediction_set) == 0:
            prediction_set = [0, 1]

        return sorted(prediction_set)

    def save(self, filepath: Path) -> None:
        """Save calibrator state to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ConformalCalibrator":
        """Load calibrator state from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class DistributionSupportStats:
    """Stores training set means and standard deviations for OOD detection."""

    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]

    def check_ood(
        self, features_dict: Dict[str, float], z_threshold: float = 3.5
    ) -> List[str]:
        """Check if any feature is significantly out of distribution."""
        warnings: List[str] = []
        for feat, val in features_dict.items():
            if feat in self.feature_means and feat in self.feature_stds:
                mean = self.feature_means[feat]
                std = max(1e-5, self.feature_stds[feat])
                z_score = abs(val - mean) / std
                if z_score > z_threshold:
                    warnings.append(
                        f"Feature '{feat}' value ({val:.2f}) is an extreme statistical outlier "
                        f"(z-score={z_score:.1f} > {z_threshold:.1f})."
                    )
        return warnings

    def save(self, filepath: Path) -> None:
        """Save training distribution stats to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(
                {
                    "feature_means": self.feature_means,
                    "feature_stds": self.feature_stds,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, filepath: Path) -> "DistributionSupportStats":
        """Load training distribution stats from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(feature_means=data["feature_means"], feature_stds=data["feature_stds"])


@dataclass
class UncertaintyAssessment:
    """Complete uncertainty evaluation for a single inference prediction."""

    calibrated_probability: float
    conformal_prediction_set: List[int]
    conformal_confidence_level: float
    decision_boundary_distance: float
    reliability_tier: str  # "HIGH", "MEDIUM", "LOW"
    reliability_explanation: str
    is_singleton_set: bool
    data_quality_warnings: List[str]


def assess_prediction_uncertainty(
    prob_class_1: float,
    raw_numeric_features: Dict[str, float],
    calibrator: ConformalCalibrator,
    support_stats: Optional[DistributionSupportStats] = None,
    extra_warnings: Optional[List[str]] = None,
) -> UncertaintyAssessment:
    """Compute holistic uncertainty, conformal sets, and reliability tier.

    Args:
        prob_class_1: Calibrated P(congestion = 1).
        raw_numeric_features: Dict of input numeric feature values.
        calibrator: Fitted ConformalCalibrator instance.
        support_stats: Optional DistributionSupportStats for OOD checks.
        extra_warnings: Warnings from prior validation stages.

    Returns:
        Structured UncertaintyAssessment.
    """
    all_warnings = list(extra_warnings or [])

    # Check for OOD
    if support_stats is not None:
        ood_warnings = support_stats.check_ood(raw_numeric_features)
        all_warnings.extend(ood_warnings)

    # Conformal prediction set
    prediction_set = calibrator.predict_set(prob_class_1)
    is_singleton = len(prediction_set) == 1
    boundary_distance = abs(prob_class_1 - 0.5)

    # Assign reliability tier
    if len(prediction_set) == 0:
        tier = "LOW"
        explanation = (
            "Empty conformal prediction set: model indicates this observation is anomalous "
            "and does not conform to either class distribution at 90% confidence."
        )
    elif len(prediction_set) == 2:
        tier = "LOW"
        explanation = (
            f"Prediction set contains both classes {prediction_set} at 90% confidence: "
            "high intrinsic operational ambiguity between congestion and normal flow."
        )
    elif len(all_warnings) > 0:
        tier = "MEDIUM"
        explanation = (
            f"Singleton prediction set {prediction_set}, but input features exhibit "
            f"data-quality/distribution warnings ({len(all_warnings)} warnings)."
        )
    elif boundary_distance < 0.15:
        tier = "MEDIUM"
        explanation = (
            f"Singleton prediction set {prediction_set}, but estimated probability ({prob_class_1:.1%}) "
            "is close to the 50% decision boundary (moderate margin)."
        )
    else:
        tier = "HIGH"
        explanation = (
            f"Singleton prediction set {prediction_set} with substantial decision margin "
            f"({prob_class_1:.1%}) and normal input telemetry support."
        )

    return UncertaintyAssessment(
        calibrated_probability=prob_class_1,
        conformal_prediction_set=prediction_set,
        conformal_confidence_level=1.0 - calibrator.alpha,
        decision_boundary_distance=float(boundary_distance),
        reliability_tier=tier,
        reliability_explanation=explanation,
        is_singleton_set=is_singleton,
        data_quality_warnings=all_warnings,
    )


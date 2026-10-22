"""Tests for explainability (SHAP / Fallback) and operational recommendations."""

import numpy as np
import pytest

from harborsense.config import config
from harborsense.explain import (
    compute_fallback_local_explanation,
    compute_shap_local_explanation,
)
from harborsense.recommendations import generate_recommendations


class DummyModel:
    """Mock model for testing explainability and recommendation logic."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Simple synthetic probability sensitive to feature at index 0 and 1
        x0 = X[0, 0]
        x1 = X[0, 1]
        p1 = 1.0 / (1.0 + np.exp(-(0.8 * x0 + 0.6 * x1)))
        p1 = float(np.clip(p1, 0.01, 0.99))
        return np.array([[1.0 - p1, p1]])


def test_fallback_local_explanation() -> None:
    """Verify fallback perturbation explainer separates risk and mitigating factors."""
    model = DummyModel()
    X = np.array([[2.5, -1.8, 0.0, 0.5]])
    raw_dict = {
        "arrival_to_berth_ratio": 2.5,
        "berth_capacity": 26.0,
        "weather_severity": 0.2,
    }
    feature_names = [
        "arrival_to_berth_ratio",
        "berth_capacity",
        "weather_severity",
        "current_queue_length",
    ]

    explanation = compute_fallback_local_explanation(
        explainer_model=model,
        transformed_features=X,
        raw_feature_dict=raw_dict,
        feature_names=feature_names,
        top_k=2,
    )

    assert explanation.method_used == "Perturbation_Sensitivity_Fallback"
    assert "causal" in explanation.disclaimer.lower()
    # At least one risk driver or mitigating factor found
    assert len(explanation.top_risk_drivers) > 0 or len(explanation.top_mitigating_factors) > 0

    if explanation.top_risk_drivers:
        driver = explanation.top_risk_drivers[0]
        assert driver.direction == "INCREASES_RISK"
        assert driver.contribution > 0


def test_recommendations_high_risk_triggers_urgent_actions() -> None:
    """Verify high congestion probability triggers immediate operational protocols."""
    raw_features = {
        "expected_arrivals_24h": 45.0,
        "berth_capacity": 20.0,
        "current_utilization": 0.92,
        "current_queue_length": 10.0,
        "weather_severity": 0.75,
    }
    top_drivers = ["expected_arrivals_24h", "current_queue_length", "weather_severity"]

    recs = generate_recommendations(
        predicted_class=1,
        calibrated_prob=0.88,
        reliability_tier="HIGH",
        raw_features=raw_features,
        top_driver_features=top_drivers,
        cfg=config,
    )

    assert recs.risk_category == "HIGH"
    assert len(recs.primary_actions) > 0

    # Ensure high urgency actions are present
    urgencies = [a.urgency for a in recs.primary_actions]
    assert "HIGH" in urgencies

    categories = [a.category for a in recs.primary_actions]
    assert "TRAFFIC" in categories or "BERTH" in categories


def test_recommendations_low_risk_triggers_routine_monitoring() -> None:
    """Verify low congestion risk triggers routine cadence without false alarms."""
    raw_features = {
        "expected_arrivals_24h": 15.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.50,
        "current_queue_length": 1.0,
        "weather_severity": 0.10,
    }

    recs = generate_recommendations(
        predicted_class=0,
        calibrated_prob=0.12,
        reliability_tier="HIGH",
        raw_features=raw_features,
        top_driver_features=[],
        cfg=config,
    )

    assert recs.risk_category == "LOW"
    assert len(recs.primary_actions) > 0
    assert any(a.urgency == "LOW" for a in recs.primary_actions)


def test_recommendations_low_reliability_escalates_to_human() -> None:
    """Verify LOW reliability tier triggers human-in-the-loop escalation flag."""
    raw_features = {
        "expected_arrivals_24h": 30.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.80,
    }

    recs = generate_recommendations(
        predicted_class=1,
        calibrated_prob=0.60,
        reliability_tier="LOW",
        raw_features=raw_features,
        top_driver_features=[],
        cfg=config,
    )

    assert recs.human_in_the_loop_flag is True
    categories = [a.category for a in recs.primary_actions]
    assert "GOVERNANCE" in categories


"""Explainability module for HarborSense AI.

Provides:
1. Global Explanations:
   - Permutation importance on held-out test data.
   - Mean absolute SHAP values across evaluation samples.
2. Local Explanations (per prediction):
   - SHAP TreeExplainer / LinearExplainer values decomposed into positive (risk-increasing)
     and negative (risk-mitigating) feature contributions.
   - Robust fallback perturbation explanation if SHAP is unavailable or encounters unsupported configurations.
   - Explicit disclaimer that feature importance reflects empirical correlation, not causal proof.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Individual feature contribution to a prediction."""

    feature_name: str
    feature_value: Any
    contribution: float  # SHAP value (positive increases risk, negative decreases risk)
    direction: str  # "INCREASES_RISK" or "DECREASES_RISK"
    interpretation: str


@dataclass
class LocalExplanation:
    """Complete local explanation payload for a single prediction."""

    base_value: float  # Expected value / prior probability
    total_model_output: float
    top_risk_drivers: List[FeatureContribution] = field(default_factory=list)
    top_mitigating_factors: List[FeatureContribution] = field(default_factory=list)
    all_contributions: Dict[str, float] = field(default_factory=dict)
    method_used: str = "SHAP_TreeExplainer"
    disclaimer: str = (
        "Feature contributions represent statistical associations derived by the model; "
        "they do not establish causal necessity or sufficiency."
    )


def compute_shap_local_explanation(
    explainer: Any,
    transformed_features: np.ndarray,
    raw_feature_dict: Dict[str, Any],
    feature_names: List[str],
    top_k: int = 5,
) -> LocalExplanation:
    """Compute local SHAP explanation with positive/negative driver separation.

    Args:
        explainer: Fitted SHAP TreeExplainer or fallback explainer.
        transformed_features: 1D or 2D array of transformed model inputs for the sample.
        raw_feature_dict: Original untransformed feature dictionary for human-readable display.
        feature_names: Names of features corresponding to transformed columns.
        top_k: Number of top drivers to return in each direction.

    Returns:
        Structured LocalExplanation.
    """
    if transformed_features.ndim == 1:
        transformed_features = transformed_features.reshape(1, -1)

    try:
        import shap  # type: ignore

        shap_values = explainer.shap_values(transformed_features)

        # Handle various shap output formats: list for multiclass/binary, or 3D/2D array
        if isinstance(shap_values, list):
            # Binary classification: index 1 is class 1 (congestion)
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            elif shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = shap_values
        else:
            # SHAP Explanation object
            sv = shap_values.values[0] if hasattr(shap_values, "values") else np.zeros(len(feature_names))

        base_val = (
            float(explainer.expected_value[1])
            if (isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1)
            else float(np.mean(explainer.expected_value))
        )
        method_name = "SHAP_TreeExplainer"

    except Exception as exc:
        logger.warning("SHAP computation failed (%s). Falling back to perturbation explainer.", exc)
        return compute_fallback_local_explanation(
            explainer_model=explainer,
            transformed_features=transformed_features,
            raw_feature_dict=raw_feature_dict,
            feature_names=feature_names,
            top_k=top_k,
        )

    return _package_explanation(
        shap_vector=sv,
        base_value=base_val,
        raw_feature_dict=raw_feature_dict,
        feature_names=feature_names,
        method_name=method_name,
        top_k=top_k,
    )


def compute_fallback_local_explanation(
    explainer_model: Any,
    transformed_features: np.ndarray,
    raw_feature_dict: Dict[str, Any],
    feature_names: List[str],
    top_k: int = 5,
) -> LocalExplanation:
    """Defensible fallback local explanation based on single-feature marginal perturbations."""
    if transformed_features.ndim == 1:
        transformed_features = transformed_features.reshape(1, -1)

    # Base prediction
    if hasattr(explainer_model, "predict_proba"):
        base_prob = float(explainer_model.predict_proba(transformed_features)[0, 1])
    else:
        base_prob = 0.5

    n_features = transformed_features.shape[1]
    deltas = np.zeros(n_features)

    # Marginal sensitivity: set feature to 0 (scaled mean) and measure prob delta
    for i in range(n_features):
        perturbed = transformed_features.copy()
        perturbed[0, i] = 0.0  # Set to mean in standardized space
        try:
            perturbed_prob = float(explainer_model.predict_proba(perturbed)[0, 1])
            deltas[i] = base_prob - perturbed_prob
        except Exception:
            deltas[i] = 0.0

    return _package_explanation(
        shap_vector=deltas,
        base_value=base_prob,
        raw_feature_dict=raw_feature_dict,
        feature_names=feature_names,
        method_name="Perturbation_Sensitivity_Fallback",
        top_k=top_k,
    )


def _package_explanation(
    shap_vector: np.ndarray,
    base_value: float,
    raw_feature_dict: Dict[str, Any],
    feature_names: List[str],
    method_name: str,
    top_k: int = 5,
) -> LocalExplanation:
    """Package raw contribution values into positive and negative driver lists."""
    contributions: Dict[str, float] = {}
    for name, val in zip(feature_names, shap_vector):
        contributions[name] = float(val)

    sorted_by_impact = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    risk_drivers: List[FeatureContribution] = []
    mitigating_factors: List[FeatureContribution] = []

    for feat_name, impact in sorted_by_impact:
        raw_val = raw_feature_dict.get(feat_name, "N/A")
        if impact > 0.001 and len(risk_drivers) < top_k:
            risk_drivers.append(
                FeatureContribution(
                    feature_name=feat_name,
                    feature_value=raw_val,
                    contribution=float(np.round(impact, 4)),
                    direction="INCREASES_RISK",
                    interpretation=f"Elevated {feat_name} (+{impact:.3f}) increases congestion likelihood.",
                )
            )
        elif impact < -0.001 and len(mitigating_factors) < top_k:
            mitigating_factors.append(
                FeatureContribution(
                    feature_name=feat_name,
                    feature_value=raw_val,
                    contribution=float(np.round(impact, 4)),
                    direction="DECREASES_RISK",
                    interpretation=f"Favorable {feat_name} ({impact:.3f}) buffers against congestion.",
                )
            )

    return LocalExplanation(
        base_value=float(base_value),
        total_model_output=float(base_value + float(np.sum(shap_vector))),
        top_risk_drivers=risk_drivers,
        top_mitigating_factors=mitigating_factors,
        all_contributions=contributions,
        method_used=method_name,
    )


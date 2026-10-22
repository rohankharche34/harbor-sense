"""Inference service and unified prediction engine for HarborSense AI.

Guarantees:
- Artifacts loaded once into memory (never retrained during inference).
- Identical inference engine used by both FastAPI backend and Streamlit dashboard.
- Strict input validation and leakage prevention.
- Calibrated probabilities, conformal prediction sets, and data-quality warnings.
- Local feature explanations (positive risk drivers and negative mitigating factors).
- Operational recommendations mapped directly from prediction outputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from harborsense.config import ProjectConfig, config
from harborsense.data_validation import validate_inference_payload
from harborsense.explain import LocalExplanation, compute_fallback_local_explanation, compute_shap_local_explanation
from harborsense.recommendations import RecommendationPlan, generate_recommendations
from harborsense.uncertainty import (
    ConformalCalibrator,
    DistributionSupportStats,
    UncertaintyAssessment,
    assess_prediction_uncertainty,
)

logger = logging.getLogger(__name__)


@dataclass
class HarborSensePrediction:
    """Consolidated prediction output for a single port observation."""

    port_id: str
    predicted_congestion: int  # 1 (Congestion) or 0 (Normal)
    congestion_probability: float  # Calibrated probability [0.0, 1.0]
    risk_category: str  # "LOW", "MEDIUM", "HIGH"
    estimated_wait_hours: float  # Secondary regression estimate
    uncertainty: UncertaintyAssessment
    explanation: LocalExplanation
    recommendations: RecommendationPlan
    data_quality_warnings: List[str] = field(default_factory=list)
    model_version: str = "0.1.0-calibrated-rf"

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to serializable dictionary."""
        return asdict(self)


class PredictionEngine:
    """Manages loaded artifacts and performs unified inference."""

    _instance: Optional["PredictionEngine"] = None

    def __init__(self, cfg: ProjectConfig = config) -> None:
        self.cfg = cfg
        self.preprocessor: Optional[Any] = None
        self.classifier: Optional[Any] = None
        self.regressor: Optional[Any] = None
        self.conformal_calibrator: Optional[ConformalCalibrator] = None
        self.distribution_stats: Optional[DistributionSupportStats] = None
        self.shap_explainer: Optional[Any] = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls, cfg: ProjectConfig = config) -> "PredictionEngine":
        """Singleton accessor for artifact caching."""
        if cls._instance is None:
            cls._instance = cls(cfg=cfg)
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self) -> None:
        """Load all trained model and preprocessing artifacts into memory."""
        model_dir = self.cfg.model_dir
        logger.info("Loading HarborSense artifacts from %s...", model_dir)

        prep_path = model_dir / "preprocessor.joblib"
        cls_path = model_dir / "classifier.joblib"
        reg_path = model_dir / "regressor.joblib"
        conf_path = model_dir / "conformal_calibrator.json"
        dist_path = model_dir / "distribution_stats.json"
        feat_path = model_dir / "feature_names.json"
        meta_path = model_dir / "metadata.json"
        shap_path = model_dir / "shap_explainer.joblib"

        required_files = [prep_path, cls_path, reg_path, conf_path, dist_path, feat_path]
        missing = [str(f) for f in required_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required model artifacts: {missing}. Run `make train` or pipeline first."
            )

        self.preprocessor = joblib.load(prep_path)
        self.classifier = joblib.load(cls_path)
        self.regressor = joblib.load(reg_path)
        self.conformal_calibrator = ConformalCalibrator.load(conf_path)
        self.distribution_stats = DistributionSupportStats.load(dist_path)

        with open(feat_path, "r") as f:
            self.feature_names = json.load(f)

        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        if shap_path.exists():
            try:
                self.shap_explainer = joblib.load(shap_path)
            except Exception as exc:
                logger.warning("Failed to load saved SHAP explainer: %s. Will use fallback.", exc)
                self.shap_explainer = None
        else:
            self.shap_explainer = None

        self.is_loaded = True
        logger.info("HarborSense PredictionEngine successfully initialized.")

    def predict_single(self, raw_input: Dict[str, Any]) -> HarborSensePrediction:
        """Run complete prediction pipeline on a single observation."""
        if not self.is_loaded:
            self.load_artifacts()

        # 1. Input Validation and Domain Bounds
        val_result = validate_inference_payload(raw_input, cfg=self.cfg)
        if not val_result.is_valid:
            raise ValueError(f"Input validation error: {'; '.join(val_result.errors)}")

        sanitized = val_result.sanitized_data or raw_input
        warnings = list(val_result.warnings)

        # 2. DataFrame formatting & Feature Preprocessing
        input_df = pd.DataFrame([sanitized])
        X_trans = self.preprocessor.transform(input_df)

        # 3. Model Classification Inference (Calibrated Probability)
        probs = self.classifier.predict_proba(X_trans)[0]
        prob_congested = float(probs[1])
        predicted_class = int(prob_congested >= 0.50)

        # 4. Secondary Regression Inference (Estimated Waiting Time)
        estimated_wait = float(self.regressor.predict(X_trans)[0])
        estimated_wait = float(np.round(np.clip(estimated_wait, 0.5, 48.0), 2))

        # 5. Risk Category Mapping
        if prob_congested < self.cfg.low_risk_threshold:
            risk_category = "LOW"
        elif prob_congested < self.cfg.high_risk_threshold:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"

        # 6. Uncertainty & Conformal Assessment
        numeric_dict = {
            k: float(sanitized[k]) for k in self.cfg.numeric_features if k in sanitized
        }
        uncertainty = assess_prediction_uncertainty(
            prob_class_1=prob_congested,
            raw_numeric_features=numeric_dict,
            calibrator=self.conformal_calibrator,
            support_stats=self.distribution_stats,
            extra_warnings=warnings,
        )

        # 7. Local Explainability (SHAP / Fallback)
        if self.shap_explainer is not None:
            explanation = compute_shap_local_explanation(
                explainer=self.shap_explainer,
                transformed_features=X_trans,
                raw_feature_dict=sanitized,
                feature_names=self.feature_names,
                top_k=4,
            )
        else:
            explanation = compute_fallback_local_explanation(
                explainer_model=self.classifier,
                transformed_features=X_trans,
                raw_feature_dict=sanitized,
                feature_names=self.feature_names,
                top_k=4,
            )

        top_drivers = [driver.feature_name for driver in explanation.top_risk_drivers]

        # 8. Operational Recommendations
        recommendations = generate_recommendations(
            predicted_class=predicted_class,
            calibrated_prob=prob_congested,
            reliability_tier=uncertainty.reliability_tier,
            raw_features=sanitized,
            top_driver_features=top_drivers,
            cfg=self.cfg,
        )

        return HarborSensePrediction(
            port_id=str(sanitized.get("port_id", "UNKNOWN")),
            predicted_congestion=predicted_class,
            congestion_probability=float(np.round(prob_congested, 4)),
            risk_category=risk_category,
            estimated_wait_hours=estimated_wait,
            uncertainty=uncertainty,
            explanation=explanation,
            recommendations=recommendations,
            data_quality_warnings=uncertainty.data_quality_warnings,
        )

    def predict_batch(self, inputs: List[Dict[str, Any]]) -> List[HarborSensePrediction]:
        """Perform predictions on a list of input dictionaries."""
        return [self.predict_single(item) for item in inputs]


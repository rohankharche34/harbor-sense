"""FastAPI backend application for HarborSense AI.

Endpoints:
- GET  /health         : Service health and model artifact loading status.
- GET  /metadata       : Model version, feature schemas, horizon, and limitations.
- GET  /metrics        : Held-out test evaluation metrics (clearly labeled offline).
- POST /predict        : Single-port real-time congestion and delay prediction.
- POST /predict/batch  : Batch prediction for multiple port observations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from harborsense.config import ProjectConfig, config
from harborsense.predict import HarborSensePrediction, PredictionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harborsense.api")

app = FastAPI(
    title="HarborSense AI - Port Congestion Prediction API",
    description=(
        "Production-grade decision-support API for 24-hour port congestion forecasting, "
        "calibrated probability estimation, conformal uncertainty assessment, "
        "explainability, and operational recommendations."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class PortTelemetryInput(BaseModel):
    """Input payload representing port operational telemetry at prediction timestamp."""

    port_id: str = Field(
        ...,
        description="Target port identifier (e.g. PORT_ROTTERDAM, PORT_SINGAPORE)",
        examples=["PORT_ROTTERDAM"],
    )
    expected_arrivals_24h: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Expected vessel arrivals over next 24 hours",
        examples=[36.0],
    )
    berth_capacity: float = Field(
        ...,
        ge=1.0,
        le=70.0,
        description="Total operational berthing positions",
        examples=[26.0],
    )
    current_utilization: float = Field(
        ...,
        ge=0.0,
        le=1.8,
        description="Current terminal berth/yard utilization ratio",
        examples=[0.88],
    )
    current_queue_length: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Number of vessels currently anchored in outer roads / waiting queue",
        examples=[7.0],
    )
    recent_arrivals_6h: float = Field(
        ...,
        ge=0.0,
        le=60.0,
        description="Number of vessel arrivals recorded in the prior 6 hours",
        examples=[11.0],
    )
    weather_severity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Adverse meteorological index (0.0 = calm, 1.0 = severe storm)",
        examples=[0.65],
    )
    historical_delay_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Past-only historical delay rate for this port (prior observations only)",
        examples=[0.32],
    )
    hour: float = Field(
        ...,
        ge=0.0,
        le=23.0,
        description="Hour of day (0-23)",
        examples=[14.0],
    )
    day_of_week: float = Field(
        ...,
        ge=0.0,
        le=6.0,
        description="Day of week (0=Monday, 6=Sunday)",
        examples=[2.0],
    )


class FeatureContributionResponse(BaseModel):
    feature_name: str
    feature_value: Any
    contribution: float
    direction: str
    interpretation: str


class ExplanationResponse(BaseModel):
    base_value: float
    method_used: str
    top_risk_drivers: List[FeatureContributionResponse]
    top_mitigating_factors: List[FeatureContributionResponse]
    disclaimer: str


class OperationalActionResponse(BaseModel):
    category: str
    urgency: str
    action: str
    rationale: str
    is_causal_guarantee: bool


class RecommendationResponse(BaseModel):
    risk_category: str
    predicted_probability: float
    primary_actions: List[OperationalActionResponse]
    contingency_actions: List[OperationalActionResponse]
    human_in_the_loop_flag: bool
    disclaimer: str


class UncertaintyResponse(BaseModel):
    calibrated_probability: float
    conformal_prediction_set: List[int]
    conformal_confidence_level: float
    decision_boundary_distance: float
    reliability_tier: str
    reliability_explanation: str
    is_singleton_set: bool
    data_quality_warnings: List[str]


class PredictionResponse(BaseModel):
    port_id: str
    predicted_congestion: int = Field(
        ..., description="Binary congestion classification (1=Congested, 0=Normal)"
    )
    congestion_probability: float = Field(
        ..., description="Calibrated probability of congestion occurring in next 24h"
    )
    risk_category: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    estimated_wait_hours: float = Field(
        ..., description="Estimated future median waiting time in hours (secondary regression task)"
    )
    uncertainty: UncertaintyResponse
    explanation: ExplanationResponse
    recommendations: RecommendationResponse
    data_quality_warnings: List[str]
    model_version: str


class BatchPredictionRequest(BaseModel):
    observations: List[PortTelemetryInput]


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    details: Dict[str, Any]


# ---------------------------------------------------------
# Lifecycle / Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup_event() -> None:
    """Initialize prediction engine on application startup."""
    try:
        engine = PredictionEngine.get_instance(config)
        logger.info("FastAPI startup: Artifacts successfully preloaded.")
    except Exception as exc:
        logger.warning(
            "FastAPI startup: Artifacts could not be loaded immediately (%s). "
            "Endpoints will attempt lazy loading.",
            exc,
        )


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health() -> HealthResponse:
    """Check system health and artifact readiness."""
    try:
        engine = PredictionEngine.get_instance(config)
        loaded = engine.is_loaded
        status_msg = "HEALTHY" if loaded else "DEGRADED"
        details = {
            "preprocessor": engine.preprocessor is not None,
            "classifier": engine.classifier is not None,
            "regressor": engine.regressor is not None,
            "conformal_calibrator": engine.conformal_calibrator is not None,
        }
    except Exception as exc:
        status_msg = "UNHEALTHY"
        loaded = False
        details = {"error": str(exc)}

    return HealthResponse(
        status=status_msg,
        models_loaded=loaded,
        details=details,
    )


@app.get("/metadata", tags=["Metadata"])
def get_metadata() -> Dict[str, Any]:
    """Retrieve model metadata, feature schemas, and operational assumptions."""
    meta_path = config.model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {"status": "Metadata file not found; train model first."}

    meta["system"] = "HarborSense AI"
    meta["prediction_horizon"] = "Next 24 Hours"
    meta["target_definition"] = f"congestion = 1 if future_median_wait_hours > {config.congestion_threshold_hours} else 0"
    meta["ports_supported"] = list(config.ports)
    meta["feature_ranges"] = config.feature_ranges
    meta["assumptions_and_limitations"] = [
        "Synthetic queue-theoretic dataset across 5 simulated global ports.",
        "A 6.0-hour waiting threshold is a designated project assumption, not an industry universal.",
        "Calibrated with sigmoid on chronological validation split; conformal coverage guaranteed marginally at 90%.",
        "Demonstration model: not validated for live marine pilotage or commercial charter operations.",
    ]
    return meta


@app.get("/metrics", tags=["Evaluation"])
def get_metrics() -> Dict[str, Any]:
    """Retrieve offline held-out test evaluation metrics.

    Note: These reflect offline test partition performance, not live streaming telemetry.
    """
    if not config.metrics_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not found. Run evaluation pipeline first.",
        )
    with open(config.metrics_file, "r") as f:
        metrics = json.load(f)

    metrics["notice"] = (
        "OFFLINE TEST RESULTS: Metrics computed exclusively on held-out chronological test partition. "
        "Do not represent real-time production SLA."
    )
    return metrics


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(input_data: PortTelemetryInput) -> PredictionResponse:
    """Predict port congestion, uncertainty, explanations, and actions for a single observation."""
    try:
        engine = PredictionEngine.get_instance(config)
        pred: HarborSensePrediction = engine.predict_single(input_data.model_dump())
        return PredictionResponse(**pred.to_dict())
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifacts unavailable: {fnf}",
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict port congestion for multiple port telemetry observations."""
    if len(request.observations) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch request contains no observations.",
        )

    try:
        engine = PredictionEngine.get_instance(config)
        dicts = [item.model_dump() for item in request.observations]
        preds = engine.predict_batch(dicts)
        return BatchPredictionResponse(
            count=len(preds),
            predictions=[PredictionResponse(**p.to_dict()) for p in preds],
        )
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {exc}",
        )


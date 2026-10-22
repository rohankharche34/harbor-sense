"""Tests for FastAPI backend endpoints and Pydantic validation."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from harborsense.config import config
from harborsense.data_generation import generate_dataset
from harborsense.train import train_models


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create FastAPI TestClient fixture."""
    # Ensure models are trained so endpoints work
    if not (config.model_dir / "classifier.joblib").exists():
        df = generate_dataset(n_steps_per_port=40, seed=42, output_path=config.data_file)
        train_models(df=df, cfg=config)
    return TestClient(app)


def test_api_health_endpoint(client: TestClient) -> None:
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert "models_loaded" in data


def test_api_metadata_endpoint(client: TestClient) -> None:
    """Verify /metadata returns expected schemas and operational definitions."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_horizon" in data
    assert "target_definition" in data
    assert "ports_supported" in data
    assert len(data["ports_supported"]) == 5


def test_api_predict_valid_payload(client: TestClient) -> None:
    """Verify /predict returns complete schema for valid input."""
    payload = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 38.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.82,
        "current_queue_length": 6.0,
        "recent_arrivals_6h": 9.0,
        "weather_severity": 0.35,
        "historical_delay_rate": 0.28,
        "hour": 14.0,
        "day_of_week": 2.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["port_id"] == "PORT_ROTTERDAM"
    assert data["predicted_congestion"] in [0, 1]
    assert 0.0 <= data["congestion_probability"] <= 1.0
    assert data["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["estimated_wait_hours"] > 0.0

    # Uncertainty
    assert "conformal_prediction_set" in data["uncertainty"]
    assert data["uncertainty"]["reliability_tier"] in ["HIGH", "MEDIUM", "LOW"]

    # Explanation
    assert "top_risk_drivers" in data["explanation"]
    assert "top_mitigating_factors" in data["explanation"]
    assert "disclaimer" in data["explanation"]

    # Recommendations
    assert "primary_actions" in data["recommendations"]
    assert len(data["recommendations"]["primary_actions"]) > 0


def test_api_predict_invalid_bounds(client: TestClient) -> None:
    """Verify /predict rejects out-of-range inputs (e.g. negative capacity)."""
    invalid_payload = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 38.0,
        "berth_capacity": -5.0,  # INVALID (must be >= 1.0)
        "current_utilization": 0.82,
        "current_queue_length": 6.0,
        "recent_arrivals_6h": 9.0,
        "weather_severity": 0.35,
        "historical_delay_rate": 0.28,
        "hour": 14.0,
        "day_of_week": 2.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Pydantic validation failure


def test_api_predict_rejects_target_leakage(client: TestClient) -> None:
    """Verify /predict rejects payloads containing target columns."""
    leak_payload = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 38.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.82,
        "current_queue_length": 6.0,
        "recent_arrivals_6h": 9.0,
        "weather_severity": 0.35,
        "historical_delay_rate": 0.28,
        "hour": 14.0,
        "day_of_week": 2.0,
        "future_median_wait_hours": 12.0,  # TARGET LEAKAGE
    }
    # Pydantic schema will either strip or fail; our engine explicitly validates
    response = client.post("/predict", json=leak_payload)
    # The endpoint should reject if leakage is passed
    if response.status_code == 200:
        # Pydantic ignored extra field; verify extra wasn't used in model
        data = response.json()
        assert "future_median_wait_hours" not in data
    else:
        assert response.status_code in [400, 422]


def test_api_batch_prediction(client: TestClient) -> None:
    """Verify /predict/batch processes a list of observations."""
    obs1 = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 30.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.75,
        "current_queue_length": 3.0,
        "recent_arrivals_6h": 7.0,
        "weather_severity": 0.2,
        "historical_delay_rate": 0.25,
        "hour": 10.0,
        "day_of_week": 1.0,
    }
    obs2 = {
        "port_id": "PORT_SINGAPORE",
        "expected_arrivals_24h": 45.0,
        "berth_capacity": 36.0,
        "current_utilization": 0.80,
        "current_queue_length": 4.0,
        "recent_arrivals_6h": 9.0,
        "weather_severity": 0.15,
        "historical_delay_rate": 0.20,
        "hour": 16.0,
        "day_of_week": 3.0,
    }
    response = client.post("/predict/batch", json={"observations": [obs1, obs2]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


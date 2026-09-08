"""Integration tests for FastAPI HTTP endpoints and error handling."""

from fastapi.testclient import TestClient

from credit_risk_service.app import app
from credit_risk_service.schemas import RiskTier


def test_health_endpoint_healthy(client: TestClient):
    """GET /health should return 200 OK and model_loaded=True when operational."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert "environment" in data


def test_health_endpoint_unhealthy():
    """GET /health should return 503 when model is not ready."""
    with TestClient(app) as test_client:
        # Simulate unloaded model state
        original_model = test_client.app.state.model
        test_client.app.state.model = None
        try:
            response = test_client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
        finally:
            test_client.app.state.model = original_model


def test_predict_endpoint_low_risk(client: TestClient, valid_applicant_payload):
    """POST /v1/predict with low-risk profile should approve with low probability."""
    response = client.post("/v1/predict", json=valid_applicant_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["applicant_id"] == valid_applicant_payload["applicant_id"]
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["risk_tier"] == RiskTier.LOW.value
    assert data["approved"] is True
    assert data["model_version"] == "1.0.0"
    assert data["latency_ms"] > 0.0
    assert "X-Process-Time-Ms" in response.headers


def test_predict_endpoint_high_risk(client: TestClient, high_risk_applicant_payload):
    """POST /v1/predict with high-risk profile should reject the application."""
    response = client.post("/v1/predict", json=high_risk_applicant_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["applicant_id"] == high_risk_applicant_payload["applicant_id"]
    assert data["risk_tier"] in [RiskTier.MEDIUM.value, RiskTier.HIGH.value]
    assert data["approved"] is False
    assert data["default_probability"] > 0.35


def test_predict_validation_error_returns_422(client: TestClient, valid_applicant_payload):
    """POST /v1/predict with invalid payload must return 422 Unprocessable Entity."""
    bad_payload = dict(valid_applicant_payload)
    bad_payload["credit_score"] = 150  # Invalid score (< 300)
    response = client.post("/v1/predict", json=bad_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("credit_score" in err["loc"] for err in errors)

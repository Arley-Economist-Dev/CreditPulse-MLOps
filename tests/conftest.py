"""Shared Pytest fixtures for unit, integration, and behavioral tests."""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from credit_risk_service.app import app
from credit_risk_service.config import settings
from credit_risk_service.model import CreditRiskModelWrapper


@pytest.fixture(scope="session")
def valid_applicant_payload() -> dict[str, Any]:
    """Standard low-risk credit applicant payload."""
    return {
        "applicant_id": "app_golden_001",
        "age": 32,
        "annual_income": 85000.0,
        "loan_amount": 12000.0,
        "credit_score": 750,
        "debt_to_income_ratio": 0.14,
        "loan_intent": "debt_consolidation",
        "employment_status": "employed",
        "historical_default": False,
    }


@pytest.fixture(scope="session")
def high_risk_applicant_payload() -> dict[str, Any]:
    """High-risk credit applicant payload."""
    return {
        "applicant_id": "app_risky_999",
        "age": 22,
        "annual_income": 18000.0,
        "loan_amount": 25000.0,
        "credit_score": 480,
        "debt_to_income_ratio": 1.38,
        "loan_intent": "personal",
        "employment_status": "unemployed",
        "historical_default": True,
    }


@pytest.fixture(scope="session")
def loaded_model() -> CreditRiskModelWrapper:
    """Pre-loaded instance of the CreditRiskModelWrapper."""
    model = CreditRiskModelWrapper(
        model_path=settings.model_path,
        decision_threshold=settings.decision_threshold,
    )
    model.load()
    return model


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient running with lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client

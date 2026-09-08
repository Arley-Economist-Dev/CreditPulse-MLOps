"""Unit tests for Pydantic v2 data contracts and schema validations."""

import pytest
from pydantic import ValidationError

from credit_risk_service.schemas import CreditRiskRequest, LoanIntent


def test_valid_credit_risk_request(valid_applicant_payload):
    """Verify that a properly structured payload passes schema validation."""
    request = CreditRiskRequest(**valid_applicant_payload)
    assert request.applicant_id == "app_golden_001"
    assert request.age == 32
    assert request.loan_intent == LoanIntent.DEBT_CONSOLIDATION
    assert request.historical_default is False


@pytest.mark.parametrize("invalid_age", [10, 17, 101, -5])
def test_age_out_of_bounds(valid_applicant_payload, invalid_age):
    """Verify age boundaries (18 <= age <= 100)."""
    payload = dict(valid_applicant_payload)
    payload["age"] = invalid_age
    with pytest.raises(ValidationError) as exc_info:
        CreditRiskRequest(**payload)
    assert "age" in str(exc_info.value)


@pytest.mark.parametrize("invalid_income", [0.0, -100.0, -0.01])
def test_income_must_be_positive(valid_applicant_payload, invalid_income):
    """Verify annual_income must be strictly greater than 0."""
    payload = dict(valid_applicant_payload)
    payload["annual_income"] = invalid_income
    with pytest.raises(ValidationError) as exc_info:
        CreditRiskRequest(**payload)
    assert "annual_income" in str(exc_info.value)


@pytest.mark.parametrize("invalid_score", [200, 299, 851, 950])
def test_credit_score_out_of_bounds(valid_applicant_payload, invalid_score):
    """Verify FICO credit_score boundaries (300 <= score <= 850)."""
    payload = dict(valid_applicant_payload)
    payload["credit_score"] = invalid_score
    with pytest.raises(ValidationError) as exc_info:
        CreditRiskRequest(**payload)
    assert "credit_score" in str(exc_info.value)


@pytest.mark.parametrize("invalid_dti", [-0.1, 5.1, 10.0])
def test_dti_ratio_out_of_bounds(valid_applicant_payload, invalid_dti):
    """Verify debt_to_income_ratio boundaries (0.0 <= dti <= 5.0)."""
    payload = dict(valid_applicant_payload)
    payload["debt_to_income_ratio"] = invalid_dti
    with pytest.raises(ValidationError) as exc_info:
        CreditRiskRequest(**payload)
    assert "debt_to_income_ratio" in str(exc_info.value)


def test_invalid_enum_rejected(valid_applicant_payload):
    """Verify that unsupported categorical strings are rejected."""
    payload = dict(valid_applicant_payload)
    payload["loan_intent"] = "cryptocurrency_speculation"
    with pytest.raises(ValidationError) as exc_info:
        CreditRiskRequest(**payload)
    assert "loan_intent" in str(exc_info.value)


def test_extra_forbidden_fields_rejected(valid_applicant_payload):
    """Verify that extra uncontracted fields are rejected (fail-fast)."""
    payload = dict(valid_applicant_payload)
    payload["unexpected_injection"] = "hacker_payload"
    with pytest.raises(ValidationError) as exc_info:
        CreditRiskRequest(**payload)
    assert "extra_forbidden" in str(exc_info.value)

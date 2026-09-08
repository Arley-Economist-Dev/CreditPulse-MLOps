"""Behavioral and metamorphic tests for the Machine Learning model.

Verifies domain-specific directional monotonicity, invariance, and business rules.
"""

from credit_risk_service.model import CreditRiskModelWrapper
from credit_risk_service.schemas import CreditRiskFeatures


def test_directional_income_sensitivity(
    loaded_model: CreditRiskModelWrapper, valid_applicant_payload
):
    """Monotonicity check: Increasing income while holding debt constant must not increase default risk."""
    base_data = dict(valid_applicant_payload)
    base_data.pop("applicant_id")

    low_income_features = CreditRiskFeatures(**{**base_data, "annual_income": 30000.0})
    high_income_features = CreditRiskFeatures(**{**base_data, "annual_income": 120000.0})

    prob_low, _, _ = loaded_model.predict_risk(low_income_features)
    prob_high, _, _ = loaded_model.predict_risk(high_income_features)

    assert prob_high <= prob_low, (
        f"Directional violation: High income ({prob_high}) has greater default prob than low income ({prob_low})"
    )


def test_directional_credit_score_sensitivity(
    loaded_model: CreditRiskModelWrapper, valid_applicant_payload
):
    """Monotonicity check: Higher credit score must result in lower or equal default probability."""
    base_data = dict(valid_applicant_payload)
    base_data.pop("applicant_id")

    poor_credit_features = CreditRiskFeatures(**{**base_data, "credit_score": 520})
    excellent_credit_features = CreditRiskFeatures(**{**base_data, "credit_score": 800})

    prob_poor, _, _ = loaded_model.predict_risk(poor_credit_features)
    prob_excellent, _, _ = loaded_model.predict_risk(excellent_credit_features)

    assert prob_excellent <= prob_poor, (
        f"Directional violation: Excellent credit ({prob_excellent}) should be <= poor credit ({prob_poor})"
    )


def test_historical_default_penalty(loaded_model: CreditRiskModelWrapper, valid_applicant_payload):
    """Applicants with prior defaults must have higher default probability than clean applicants."""
    base_data = dict(valid_applicant_payload)
    base_data.pop("applicant_id")

    clean_features = CreditRiskFeatures(**{**base_data, "historical_default": False})
    defaulter_features = CreditRiskFeatures(**{**base_data, "historical_default": True})

    prob_clean, _, _ = loaded_model.predict_risk(clean_features)
    prob_defaulter, _, _ = loaded_model.predict_risk(defaulter_features)

    assert prob_defaulter > prob_clean, (
        f"Default history penalty violation: {prob_defaulter} not strictly greater than {prob_clean}"
    )


def test_prediction_invariance(loaded_model: CreditRiskModelWrapper, valid_applicant_payload):
    """Evaluating identical input multiple times must yield exact deterministic outputs."""
    base_data = dict(valid_applicant_payload)
    base_data.pop("applicant_id")
    features = CreditRiskFeatures(**base_data)

    prob_1, tier_1, approved_1 = loaded_model.predict_risk(features)
    prob_2, tier_2, approved_2 = loaded_model.predict_risk(features)

    assert prob_1 == prob_2
    assert tier_1 == tier_2
    assert approved_1 == approved_2


def test_decision_threshold_boundary_consistency(
    loaded_model: CreditRiskModelWrapper, valid_applicant_payload
):
    """Approval status must be strictly consistent with decision_threshold."""
    base_data = dict(valid_applicant_payload)
    base_data.pop("applicant_id")
    features = CreditRiskFeatures(**base_data)

    prob, _, approved = loaded_model.predict_risk(features)
    if prob < loaded_model.decision_threshold:
        assert approved is True
    else:
        assert approved is False

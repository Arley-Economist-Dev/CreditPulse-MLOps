"""Data contracts and schemas for the Credit Risk Inference Service."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class LoanIntent(str, Enum):
    """Permitted purposes for the loan."""

    PERSONAL = "personal"
    EDUCATION = "education"
    MEDICAL = "medical"
    VENTURE = "venture"
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"


class EmploymentStatus(str, Enum):
    """Employment category of the applicant."""

    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"


class RiskTier(str, Enum):
    """Categorical risk classification based on predicted probability."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CreditRiskFeatures(BaseModel):
    """Features required by the credit risk ML model."""

    model_config = ConfigDict(extra="forbid")

    age: Annotated[
        int,
        Field(
            ge=18,
            le=100,
            description="Applicant age in years (must be between 18 and 100).",
            examples=[34],
        ),
    ]
    annual_income: Annotated[
        float,
        Field(
            gt=0.0,
            le=10_000_000.0,
            description="Annual gross income in USD.",
            examples=[55000.0],
        ),
    ]
    loan_amount: Annotated[
        float,
        Field(
            gt=0.0,
            le=1_000_000.0,
            description="Requested loan principal amount in USD.",
            examples=[15000.0],
        ),
    ]
    credit_score: Annotated[
        int,
        Field(
            ge=300,
            le=850,
            description="FICO credit score between 300 and 850.",
            examples=[720],
        ),
    ]
    debt_to_income_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=5.0,
            description="Debt-to-Income (DTI) ratio, typically between 0.0 and 5.0.",
            examples=[0.27],
        ),
    ]
    loan_intent: Annotated[
        LoanIntent,
        Field(
            description="Declared purpose for the loan.",
            examples=[LoanIntent.DEBT_CONSOLIDATION],
        ),
    ]
    employment_status: Annotated[
        EmploymentStatus,
        Field(
            description="Employment status of the applicant.",
            examples=[EmploymentStatus.EMPLOYED],
        ),
    ]
    historical_default: Annotated[
        bool,
        Field(
            description="True if the applicant previously defaulted on a loan.",
            examples=[False],
        ),
    ]


class CreditRiskRequest(CreditRiskFeatures):
    """Payload for real-time inference requests."""

    applicant_id: Annotated[
        str,
        Field(
            min_length=3,
            max_length=64,
            pattern=r"^[a-zA-Z0-9_-]+$",
            description="Unique identifier for the credit applicant.",
            examples=["app_98234"],
        ),
    ]


class CreditRiskResponse(BaseModel):
    """Structured response from the inference service."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    default_probability: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Predicted probability of loan default."),
    ]
    risk_tier: RiskTier
    approved: bool
    model_version: str
    latency_ms: Annotated[float, Field(ge=0.0, description="Inference latency in milliseconds.")]

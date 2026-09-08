"""Model loading and real-time prediction wrapper."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_risk_service.schemas import CreditRiskFeatures, RiskTier

logger = logging.getLogger(__name__)


class CreditRiskModelWrapper:
    """Encapsulates model lifecycle, feature alignment, and inference."""

    def __init__(self, model_path: Path, decision_threshold: float = 0.35) -> None:
        self.model_path = model_path
        self.decision_threshold = decision_threshold
        self._pipeline: Any = None
        self._metadata: dict[str, Any] = {}
        self.version: str = "unknown"

    @property
    def is_loaded(self) -> bool:
        """True if the model artifact is loaded into memory."""
        return self._pipeline is not None

    def load(self) -> None:
        """Load the serialized scikit-learn pipeline into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Ensure DVC has pulled the model before starting the service."
            )

        logger.info("Loading model artifact from %s", self.model_path)
        self._pipeline = joblib.load(self.model_path)

        metadata_path = self.model_path.parent / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                    self.version = self._metadata.get("version", "1.0.0")
            except Exception as e:
                logger.warning("Could not read metadata from %s: %s", metadata_path, e)
                self.version = "1.0.0"
        else:
            self.version = "1.0.0"

        logger.info("Model loaded successfully. Version: %s", self.version)

    def predict_risk(self, features: CreditRiskFeatures) -> tuple[float, RiskTier, bool]:
        """Perform real-time inference on applicant features.

        Returns:
            tuple[float, RiskTier, bool]: (default_probability, risk_tier, approved)
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() before predicting.")

        # Construct single-row DataFrame aligned with training schema
        df = pd.DataFrame(
            [
                {
                    "age": features.age,
                    "annual_income": features.annual_income,
                    "loan_amount": features.loan_amount,
                    "credit_score": features.credit_score,
                    "debt_to_income_ratio": features.debt_to_income_ratio,
                    "loan_intent": features.loan_intent.value,
                    "employment_status": features.employment_status.value,
                    "historical_default": int(features.historical_default),
                }
            ]
        )

        probabilities = self._pipeline.predict_proba(df)
        default_prob = round(float(probabilities[0, 1]), 4)

        if default_prob < 0.20:
            risk_tier = RiskTier.LOW
        elif default_prob < 0.45:
            risk_tier = RiskTier.MEDIUM
        else:
            risk_tier = RiskTier.HIGH

        approved = bool(default_prob < self.decision_threshold)
        return default_prob, risk_tier, approved

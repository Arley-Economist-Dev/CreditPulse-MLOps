"""Deterministic training pipeline for Credit Risk Baseline Model.

Generates a realistic synthetic credit risk dataset based on business rules,
trains a scikit-learn Pipeline with proper preprocessing, evaluates performance,
and serializes the resulting model artifact and evaluation metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk_service.schemas import EmploymentStatus, LoanIntent

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "debt_to_income_ratio",
]

CATEGORICAL_FEATURES = [
    "loan_intent",
    "employment_status",
]

BOOLEAN_FEATURES = [
    "historical_default",
]

ALL_FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def generate_synthetic_credit_data(
    n_samples: int = 5000, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate realistic synthetic credit applicant data with business ground truth."""
    rng = np.random.default_rng(random_state)

    ages = rng.integers(18, 75, size=n_samples)
    annual_incomes = np.round(rng.lognormal(mean=10.8, sigma=0.6, size=n_samples), -2)
    # Clip incomes between 15,000 and 350,000
    annual_incomes = np.clip(annual_incomes, 15_000, 350_000)

    # Loan amount depends somewhat on income
    loan_amounts = np.round(
        rng.uniform(1_000, 45_000, size=n_samples) * (annual_incomes / 60_000).clip(0.5, 1.8),
        -2,
    )
    loan_amounts = np.clip(loan_amounts, 1_000, 50_000)

    credit_scores = rng.integers(350, 850, size=n_samples)
    dti_ratios = np.round(loan_amounts / (annual_incomes + 1e-5), 3)

    loan_intents = rng.choice(
        [e.value for e in LoanIntent],
        size=n_samples,
        p=[0.30, 0.20, 0.15, 0.10, 0.15, 0.10],
    )

    employment_statuses = rng.choice(
        [e.value for e in EmploymentStatus],
        size=n_samples,
        p=[0.75, 0.18, 0.07],
    )

    # Historical default is more likely for low credit scores
    hist_default_prob = np.clip((700 - credit_scores) / 500, 0.02, 0.60)
    historical_defaults = rng.random(size=n_samples) < hist_default_prob

    df = pd.DataFrame(
        {
            "age": ages,
            "annual_income": annual_incomes,
            "loan_amount": loan_amounts,
            "credit_score": credit_scores,
            "debt_to_income_ratio": dti_ratios,
            "loan_intent": loan_intents,
            "employment_status": employment_statuses,
            "historical_default": historical_defaults.astype(int),
        }
    )

    # Log-odds calculation for ground-truth default probability
    # Base risk
    log_odds = (
        -1.5
        - (df["credit_score"] - 650) / 75.0
        + (df["debt_to_income_ratio"] - 0.25) * 3.5
        - (df["annual_income"] - 50_000) / 45_000.0
        + (df["loan_amount"] - 15_000) / 20_000.0
        + df["historical_default"] * 1.8
        + (df["employment_status"] == EmploymentStatus.UNEMPLOYED.value) * 1.2
    )

    probs = 1.0 / (1.0 + np.exp(-log_odds))
    y = (rng.random(size=n_samples) < probs).astype(int)

    return df, pd.Series(y, name="default")


def build_pipeline() -> Pipeline:
    """Construct an end-to-end preprocessing and classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_export(output_dir: Path) -> dict[str, float]:
    """Train the model, evaluate metrics, and serialize artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "credit_risk_model.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"

    print("Generating synthetic credit dataset...")
    x_df, y_series = generate_synthetic_credit_data(n_samples=6000, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_series, test_size=0.20, random_state=42, stratify=y_series
    )

    print(f"Training set: {len(x_train)} samples | Test set: {len(x_test)} samples")
    pipeline = build_pipeline()

    print("Fitting Pipeline (StandardScaler + OneHotEncoder + HistGradientBoosting)...")
    pipeline.fit(x_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(x_test)
    y_pred_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    }

    print("Evaluation Results on Holdout Set:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Serialize Model Artifact
    joblib.dump(pipeline, model_path, compress=3)
    print(f"Model saved to: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # Save Metrics
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save Metadata
    metadata = {
        "model_name": "credit_risk_hist_gradient_boosting",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn",
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "boolean": BOOLEAN_FEATURES,
        },
        "metrics": metrics,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_and_export(project_root / "models")

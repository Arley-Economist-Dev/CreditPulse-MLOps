"""Configuration settings for Credit Risk Inference Service."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime service configuration loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore",
    )

    app_name: str = "Credit Risk Real-Time Inference Service"
    app_version: str = "0.1.0"
    environment: str = "production"
    log_level: str = "INFO"

    # Model parameters
    model_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"
    model_filename: str = "credit_risk_model.joblib"
    decision_threshold: float = 0.35  # Approvals require default_probability < threshold

    @property
    def model_path(self) -> Path:
        """Full path to serialized model binary."""
        return self.model_dir / self.model_filename


settings = Settings()

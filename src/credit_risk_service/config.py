"""Configuration settings for Credit Risk Inference Service."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_model_dir() -> Path:
    """Resolve models directory across local, editable, and container environments."""
    candidates = [
        Path.cwd() / "models",
        Path("/app/models"),
        Path(__file__).resolve().parent.parent.parent / "models",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("models")


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
    model_dir: Path = _resolve_model_dir()
    model_filename: str = "credit_risk_model.joblib"
    decision_threshold: float = 0.35  # Approvals require default_probability < threshold

    @property
    def model_path(self) -> Path:
        """Full path to serialized model binary."""
        return self.model_dir / self.model_filename


settings = Settings()

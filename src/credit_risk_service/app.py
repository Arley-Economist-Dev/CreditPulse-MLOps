"""FastAPI microservice for Real-Time Credit Risk Inference."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from credit_risk_service.config import settings
from credit_risk_service.model import CreditRiskModelWrapper
from credit_risk_service.schemas import CreditRiskRequest, CreditRiskResponse

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("credit_risk_service")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown lifecycle events.

    Loads the machine learning model into application state during startup,
    ensuring zero latency overhead on the first incoming request.
    """
    logger.info("Initializing Credit Risk Inference Service...")
    model_wrapper = CreditRiskModelWrapper(
        model_path=settings.model_path,
        decision_threshold=settings.decision_threshold,
    )

    try:
        model_wrapper.load()
        logger.info("Model ready for inference. Version: %s", model_wrapper.version)
    except Exception as exc:
        logger.error("Failed to load model during startup: %s", exc)

    app.state.model = model_wrapper
    yield
    logger.info("Shutting down Credit Risk Inference Service.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-grade microservice for low-latency credit risk scoring. "
        "Built with FastAPI, Pydantic v2, and scikit-learn."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """Measure total request processing time and attach header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


@app.get(
    "/health",
    tags=["Observability"],
    summary="Health and Readiness Probe",
    responses={
        200: {"description": "Service healthy and model ready to accept traffic."},
        503: {"description": "Model is not loaded; service cannot serve predictions."},
    },
)
async def health_check() -> JSONResponse:
    """Liveness and readiness check for orchestrators (Kubernetes / ECS)."""
    model_wrapper: CreditRiskModelWrapper = getattr(app.state, "model", None)
    is_ready = model_wrapper is not None and model_wrapper.is_loaded

    payload = {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_version": model_wrapper.version if is_ready else None,
        "environment": settings.environment,
    }

    if not is_ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)

    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


@app.post(
    "/v1/predict",
    response_model=CreditRiskResponse,
    tags=["Inference"],
    summary="Evaluate Credit Risk in Real-Time",
    status_code=status.HTTP_200_OK,
)
async def predict_credit_risk(request: CreditRiskRequest) -> CreditRiskResponse:
    """Predict default probability and risk tier for an applicant."""
    model_wrapper: CreditRiskModelWrapper = getattr(app.state, "model", None)

    if model_wrapper is None or not model_wrapper.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference.",
        )

    t0 = time.perf_counter()
    try:
        default_prob, risk_tier, approved = model_wrapper.predict_risk(request)
    except Exception as err:
        logger.exception("Inference error for applicant %s: %s", request.applicant_id, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference failure.",
        ) from err

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return CreditRiskResponse(
        applicant_id=request.applicant_id,
        default_probability=default_prob,
        risk_tier=risk_tier,
        approved=approved,
        model_version=model_wrapper.version,
        latency_ms=latency_ms,
    )

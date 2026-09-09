# syntax=docker/dockerfile:1
# ------------------------------------------------------------------------------
# Production Dockerfile for Credit Risk Real-Time Inference Service
# Minimal footprint (python:3.11-slim), non-root execution (UID 10001)
# ------------------------------------------------------------------------------

FROM python:3.11-slim as runtime

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Create unprivileged application user and group (Principle of Least Privilege)
RUN addgroup --system --gid 10001 appgroup && \
    adduser --system --uid 10001 --gid 10001 --no-create-home appuser

WORKDIR /app

# Copy dependency specifications first for Docker layer caching
COPY pyproject.toml README.md /app/

# Install only production dependencies (excluding dev and test)
RUN pip install --no-cache-dir .

# Copy application source code, frontend dashboard and serialized model artifacts
COPY src/ /app/src/
COPY frontend/ /app/frontend/
COPY models/credit_risk_model.joblib /app/models/credit_risk_model.joblib
COPY models/metadata.json /app/models/metadata.json

# Enforce secure ownership and switch to non-root user
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Native healthcheck using Python standard library (no curl/wget bloat needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode() == 200 else 1)"

# Production ASGI server launch
CMD ["uvicorn", "credit_risk_service.app:app", "--host", "0.0.0.0", "--port", "8000"]

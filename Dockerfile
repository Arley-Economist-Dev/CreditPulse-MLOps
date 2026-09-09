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

# Copy package definitions, source code, and assets
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY frontend/ /app/frontend/
COPY scripts/ /app/scripts/
COPY models/ /app/models/

# Install production dependencies and the package
RUN pip install --no-cache-dir .

# Ensure model binary artifact is present in the image
# If the heavy binary was not stored in Git due to DVC, train baseline deterministically
RUN python scripts/train_baseline.py

# Enforce secure ownership and switch to non-root user
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Native healthcheck using Python standard library (supports dynamic $PORT)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys, os; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{port}/health').getcode() == 200 else 1)"

# Production ASGI server launch supporting Render dynamic $PORT
CMD ["sh", "-c", "uvicorn credit_risk_service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

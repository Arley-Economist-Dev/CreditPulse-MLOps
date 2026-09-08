.PHONY: help install lint format test train run docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install package in editable mode with dev & test dependencies"
	@echo "  make lint         - Run Ruff code linter and formatting check"
	@echo "  make format       - Apply automated formatting with Ruff"
	@echo "  make test         - Run full test suite with coverage report"
	@echo "  make train        - Execute deterministic baseline training pipeline"
	@echo "  make run          - Launch FastAPI development server"
	@echo "  make docker-build - Build production hardened Docker image"
	@echo "  make docker-run   - Run container locally on port 8000"
	@echo "  make clean        - Remove caches and temporary files"

install:
	pip install --upgrade pip
	pip install -e ".[dev,test]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -v --cov=credit_risk_service --cov-report=term-missing

train:
	python scripts/train_baseline.py

run:
	uvicorn credit_risk_service.app:app --reload --port 8000

docker-build:
	docker build -t credit-risk-service:latest .

docker-run:
	docker run -d -p 8000:8000 --name credit-risk-api credit-risk-service:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

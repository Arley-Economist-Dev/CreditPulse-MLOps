# Credit Risk Real-Time Inference Service

[![CI Pipeline](https://github.com/your-org/credit-risk-service/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/credit-risk-service/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Microservicio de inferencia de Machine Learning de baja latencia con contratos estrictos (FastAPI + Pydantic v2), versionado desacoplado de artefactos (DVC), testing automatizado (Pytest) y contenedorización segura para producción (Docker + GitHub Actions).

---

## 🏛️ Arquitectura del Repositorio

```text
MLOPS/
├── .github/workflows/         # Pipelines de CI/CD automatizados
├── data/
│   ├── raw/                   # Datos crudos rastreados por DVC
│   └── processed/             # Datos procesados / datasets de evaluación
├── models/                    # Pesos y artefactos serializados rastreados por DVC (.dvc)
├── src/
│   └── credit_risk_service/   # Código fuente modular (src layout)
│       ├── __init__.py
│       ├── config.py          # Parámetros y settings de runtime
│       ├── schemas.py         # Contratos Pydantic v2 (Request / Response)
│       ├── model.py           # Envoltorio e inferencia del modelo
│       └── app.py             # FastAPI entrypoint y middleware
├── tests/                     # Batería de pruebas automatizadas
│   ├── unit/                  # Tests unitarios de preprocesamiento
│   ├── integration/           # Tests de endpoints HTTP y contratos de API
│   └── behavioral/            # Tests de invarianza y direccionalidad del modelo
├── .gitignore                 # Exclusión de binarios y archivos temporales
├── pyproject.toml             # Gestión de dependencias y configuración de herramientas
└── README.md
```

---

## 🚀 Inicio Rápido (Desarrollo Local)

### 1. Requisitos Previos
* Python `>= 3.10`
* Git

### 2. Configuración del Entorno Virtual
```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Actualizar pip e instalar en modo editable con herramientas de desarrollo y test
pip install --upgrade pip
pip install -e ".[dev,test]"
```

### 3. Calidad de Código (Linting & Formateo)
```powershell
# Comprobar linting
ruff check .

# Comprobar formato
ruff format --check .

# Aplicar auto-formato
ruff format .
```

### 4. Ejecución de Tests
```powershell
pytest
```

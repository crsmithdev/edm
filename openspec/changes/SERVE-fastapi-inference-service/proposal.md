---
status: ready
created: 2025-12-05
---

# [SERVE] Build FastAPI Inference Service

## Why

**No Deployment Infrastructure**: CLI-only interface, cannot serve models in production or integrate with other systems.

**Requirements**:
- REST API for predictions
- Load model from MLflow registry
- Health checks and monitoring hooks
- Containerization for deployment

**Impact**: Enables production use, horizontal scaling, integration with other services.

## What

- Create `src/edm/serving/api.py` - FastAPI application
- Create `Dockerfile` - Containerized deployment
- Create `docker-compose.yml` - Local dev environment
- Endpoints: `/predict`, `/health`, `/model/info`

## Impact

- **Effort**: 4 days
- **ROI**: High - critical for production
- **Dependencies**: FastAPI, uvicorn, Docker
- **Prerequisite**: MLREG (MLflow registry)

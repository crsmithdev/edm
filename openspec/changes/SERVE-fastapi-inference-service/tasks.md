# Tasks

## 1. Setup FastAPI
- [ ] 1.1 Add `fastapi>=0.110.0`, `uvicorn>=0.27.0` to dependencies
- [ ] 1.2 Create `src/edm/serving/` directory
- [ ] 1.3 Create `src/edm/serving/api.py`

## 2. Implement Endpoints
- [ ] 2.1 `POST /predict` - Audio file upload → predictions
- [ ] 2.2 `GET /health` - Service health check
- [ ] 2.3 `GET /model/info` - Current model metadata
- [ ] 2.4 `GET /metrics` - Prometheus metrics endpoint

## 3. Model Loading
- [ ] 3.1 Load production model from MLflow on startup
- [ ] 3.2 Implement hot-reloading for model updates
- [ ] 3.3 Add model warmup (test prediction on startup)

## 4. Containerization
- [ ] 4.1 Create `Dockerfile`
- [ ] 4.2 Create `docker-compose.yml` (API + MLflow)
- [ ] 4.3 Test local deployment
- [ ] 4.4 Add `.dockerignore`

## 5. Testing
- [ ] 5.1 Integration tests for all endpoints
- [ ] 5.2 Load testing with locust
- [ ] 5.3 Test model inference accuracy

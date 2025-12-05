# Tasks

## 1. Add Prometheus Metrics
- [ ] 1.1 Add `prometheus-client>=0.19.0` dependency
- [ ] 1.2 Create `src/edm/serving/monitoring.py`
- [ ] 1.3 Define metrics: predictions_total, latency, confidence, errors, gpu_memory
- [ ] 1.4 Wrap inference with monitoring

## 2. Integrate with API
- [ ] 2.1 Add `/metrics` endpoint to FastAPI
- [ ] 2.2 Update `/predict` to record metrics
- [ ] 2.3 Record per-request latency and confidence

## 3. Grafana Setup
- [ ] 3.1 Add Grafana to `docker-compose.yml`
- [ ] 3.2 Create dashboard JSON config
- [ ] 3.3 Add panels: predictions/sec, P95 latency, confidence distribution, error rate

## 4. Alerting
- [ ] 4.1 Create `alerts.yml` with Prometheus rules
- [ ] 4.2 Alert: High error rate (>5%)
- [ ] 4.3 Alert: Low confidence (median <0.7)
- [ ] 4.4 Alert: High latency (P95 >5s)

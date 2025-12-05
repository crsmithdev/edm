---
status: ready
created: 2025-12-05
---

# [MONITOR] Add Prometheus + Grafana Monitoring

## Why

**No Production Observability**: Cannot track inference latency, error rates, model confidence, or resource usage.

**Requirements**:
- Real-time metrics collection
- Dashboards for visualization
- Alerting on degradation
- SLA tracking

**Impact**: Detect model issues before users report them, capacity planning, performance optimization.

## What

- Extend `src/edm/serving/api.py` with Prometheus metrics
- Create `src/edm/serving/monitoring.py` - Monitored inference wrapper
- Create Grafana dashboard config
- Create alert rules (high error rate, low confidence, high latency)

## Impact

- **Effort**: 3 days
- **ROI**: High - essential for production
- **Dependencies**: prometheus-client, Grafana (Docker)
- **Prerequisite**: SERVE (FastAPI service)

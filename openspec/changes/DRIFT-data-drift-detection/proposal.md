---
status: ready
created: 2025-12-05
---

# [DRIFT] Implement Data Drift Detection

## Why

**No Drift Detection**: Models degrade over time as production data distribution shifts (new EDM subgenres, production styles).

**Requirements**:
- Monitor BPM, energy, duration distributions
- Compare production vs training distributions
- Alert on significant drift
- Trigger retraining automatically

**Impact**: Maintain model accuracy over time, proactive retraining, prevent silent degradation.

## What

- Create `src/edm/monitoring/drift.py` - Drift detection module
- Implement `DriftDetector` (KS-test, Wasserstein distance)
- Implement `DriftMonitor` (buffer production samples, scheduled checks)
- Integrate with serving API
- Compute reference statistics from training data

## Impact

- **Effort**: 3 days
- **ROI**: High - maintains accuracy
- **Dependencies**: scipy
- **Prerequisite**: SERVE (need production predictions)

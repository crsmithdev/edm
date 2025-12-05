# Tasks

## 1. Create Drift Detection
- [ ] 1.1 Create `src/edm/monitoring/drift.py`
- [ ] 1.2 Implement `DriftDetector` class
- [ ] 1.3 Add KS-test for distribution comparison
- [ ] 1.4 Add Wasserstein distance for severity
- [ ] 1.5 Implement `generate_recommendation()` logic

## 2. Reference Statistics
- [ ] 2.1 Compute training data statistics (BPM, energy, duration)
- [ ] 2.2 Save to `data/reference_stats.npz`
- [ ] 2.3 Load on drift detector initialization

## 3. Production Monitoring
- [ ] 3.1 Implement `DriftMonitor` class
- [ ] 3.2 Buffer production predictions
- [ ] 3.3 Scheduled drift checks (daily)
- [ ] 3.4 Alert on drift detection

## 4. Integration
- [ ] 4.1 Log predictions in serving API
- [ ] 4.2 Background task for periodic checks
- [ ] 4.3 Alert integration (email/Slack)
- [ ] 4.4 Optional: Trigger retraining on critical drift

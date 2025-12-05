# Tasks

## 1. Setup MLflow
- [ ] 1.1 Add `mlflow>=2.10.0` to `pyproject.toml`
- [ ] 1.2 Create `mlruns/` directory (gitignored)
- [ ] 1.3 Test MLflow UI: `mlflow ui`

## 2. Create Registry Module
- [ ] 2.1 Create `src/edm/registry/mlflow_registry.py`
- [ ] 2.2 Implement `ModelRegistry` class
- [ ] 2.3 Add `log_model()` method
- [ ] 2.4 Add `promote_model()` method
- [ ] 2.5 Add `load_production_model()` method

## 3. Integrate with Trainer
- [ ] 3.1 Update `Trainer.__init__()` to accept registry
- [ ] 3.2 Call `registry.log_model()` after best model saved
- [ ] 3.3 Log hyperparameters, metrics, git commit

## 4. CLI Integration
- [ ] 4.1 Add `edm models list` command
- [ ] 4.2 Add `edm models promote <version> <stage>` command
- [ ] 4.3 Add `edm models info <version>` command

## 5. Documentation
- [ ] 5.1 Add MLflow guide to `docs/mlops.md`
- [ ] 5.2 Document promotion workflow
- [ ] 5.3 Add examples to README

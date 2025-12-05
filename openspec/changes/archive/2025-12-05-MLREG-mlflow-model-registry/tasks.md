# Tasks

## 1. Setup MLflow
- [x] 1.1 Add `mlflow>=2.10.0` to `pyproject.toml`
- [x] 1.2 Create `mlruns/` directory (gitignored)
- [x] 1.3 Test MLflow UI: `mlflow ui`

## 2. Create Registry Module
- [x] 2.1 Create `src/edm/registry/mlflow_registry.py`
- [x] 2.2 Implement `ModelRegistry` class
- [x] 2.3 Add `log_model()` method
- [x] 2.4 Add `promote_model()` method
- [x] 2.5 Add `load_production_model()` method

## 3. Integrate with Trainer
- [x] 3.1 Update `Trainer.__init__()` to accept registry
- [x] 3.2 Call `registry.log_model()` after best model saved
- [x] 3.3 Log hyperparameters, metrics, git commit

## 4. CLI Integration
- [x] 4.1 Add `edm models list` command
- [x] 4.2 Add `edm models promote <version> <stage>` command
- [x] 4.3 Add `edm models info <version>` command

## 5. Documentation
- [x] 5.1 Add MLflow guide to `docs/mlops.md`
- [x] 5.2 Document promotion workflow
- [x] 5.3 Add examples to README

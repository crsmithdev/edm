"""MLflow model registry integration."""

import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistry:
    """Wrapper for MLflow model registry operations."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "edm-training",
    ):
        """Initialize model registry.

        Args:
            tracking_uri: MLflow tracking URI (default: ./mlruns)
            experiment_name: MLflow experiment name
        """
        self.tracking_uri = tracking_uri or "./mlruns"
        self.experiment_name = experiment_name

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.experiment = mlflow.set_experiment(experiment_name)

        # Initialize client
        self.client = MlflowClient()

    def log_model(
        self,
        model_path: Path,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> str:
        """Log trained model to MLflow.

        Args:
            model_path: Path to saved model checkpoint (.pt file)
            run_name: Name for this training run
            params: Training hyperparameters
            metrics: Final metrics (val_loss, etc.)
            tags: Additional tags

        Returns:
            MLflow run ID
        """
        # Get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_commit = "unknown"

        # Start run
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            default_tags = {
                "git_commit": git_commit,
                "model_type": "multitask",
            }
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            # Log model artifact
            mlflow.log_artifact(str(model_path), artifact_path="model")

            # Log model to registry
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="edm-structure-detector",
            )

            print(f"Model registered: {registered_model.name} v{registered_model.version}")
            print(f"Run ID: {run.info.run_id}")

            return str(run.info.run_id)

    def promote_model(self, version: int, stage: str) -> None:
        """Promote model to a specific stage.

        Args:
            version: Model version number
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")

        self.client.transition_model_version_stage(
            name="edm-structure-detector",
            version=version,
            stage=stage,
        )

        print(f"Model version {version} promoted to {stage}")

    def load_production_model(self) -> Path | None:
        """Get path to current production model.

        Returns:
            Path to production model checkpoint, or None if no production model exists
        """
        try:
            # Get latest production version
            versions = self.client.get_latest_versions(
                name="edm-structure-detector",
                stages=["Production"],
            )

            if not versions:
                return None

            latest = versions[0]
            run_id = latest.run_id

            # Get artifact location
            run = self.client.get_run(run_id)
            artifact_uri = run.info.artifact_uri

            # Download artifact
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{artifact_uri}/model",
                dst_path="./downloads",
            )

            return Path(model_path)

        except Exception as e:
            print(f"Error loading production model: {e}")
            return None

    def list_models(self, max_results: int = 10) -> list[dict[str, Any]]:
        """List registered models with metadata.

        Args:
            max_results: Maximum number of results to return

        Returns:
            List of model metadata dicts
        """
        models = []

        # Get all model versions
        versions = self.client.search_model_versions(
            filter_string="name='edm-structure-detector'",
            max_results=max_results,
            order_by=["version_number DESC"],
        )

        for version in versions:
            run = self.client.get_run(version.run_id)

            models.append(
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "unknown"),
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "created_at": version.creation_timestamp,
                }
            )

        return models

    def get_model_info(self, version: int) -> dict[str, Any] | None:
        """Get detailed info for a specific model version.

        Args:
            version: Model version number

        Returns:
            Model info dict, or None if not found
        """
        try:
            version_info = self.client.get_model_version(
                name="edm-structure-detector",
                version=str(version),
            )

            run = self.client.get_run(version_info.run_id)

            return {
                "version": version_info.version,
                "stage": version_info.current_stage,
                "run_id": version_info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "unknown"),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "created_at": version_info.creation_timestamp,
                "description": version_info.description,
            }

        except Exception as e:
            print(f"Model version {version} not found: {e}")
            return None

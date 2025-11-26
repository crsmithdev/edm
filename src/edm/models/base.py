"""Base model classes and loading utilities."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BaseModel(ABC):
    """Base class for ML models."""

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make a prediction.

        Args:
            data: Input data for the model.

        Returns:
            Model prediction.
        """
        pass


def load_model(model_name: str, _model_path: Path | None = None) -> BaseModel:
    """Load a pre-trained model.

    Args:
        model_name: Name or identifier of the model.
        model_path: Custom path to model file.

    Returns:
        Loaded model instance.

    Raises:
        ModelNotFoundError: If the model cannot be found.

    Examples:
        >>> model = load_model("bpm-detector")
        >>> prediction = model.predict(audio_features)
    """
    logger.info("loading model", model_name=model_name)

    # TODO: Implement actual model loading
    from edm.exceptions import ModelNotFoundError

    raise ModelNotFoundError(f"Model '{model_name}' not implemented yet")

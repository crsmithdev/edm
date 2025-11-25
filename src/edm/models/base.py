"""Base model classes and loading utilities."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for ML models."""

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make a prediction.

        Parameters
        ----------
        data : Any
            Input data for the model.

        Returns
        -------
        Any
            Model prediction.
        """
        pass


def load_model(model_name: str, _model_path: Optional[Path] = None) -> BaseModel:
    """Load a pre-trained model.

    Parameters
    ----------
    model_name : str
        Name or identifier of the model.
    model_path : Path, optional
        Custom path to model file.

    Returns
    -------
    BaseModel
        Loaded model instance.

    Raises
    ------
    ModelNotFoundError
        If the model cannot be found.

    Examples
    --------
    >>> model = load_model("bpm-detector")
    >>> prediction = model.predict(audio_features)
    """
    logger.info(f"Loading model: {model_name}")

    # TODO: Implement actual model loading
    from edm.exceptions import ModelNotFoundError

    raise ModelNotFoundError(f"Model '{model_name}' not implemented yet")

"""Pretrained backbone models for feature extraction."""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel


class MERTBackbone(nn.Module):
    """MERT (Music Understanding Model with Large-Scale Self-supervised Training) backbone.

    MERT is a pretrained transformer model for music understanding tasks.
    We use it as a feature extractor, optionally fine-tuning the last few layers.

    Attributes:
        model_name: HuggingFace model identifier
        freeze_layers: Number of encoder layers to freeze (from bottom)
        device: Torch device (cuda/cpu)
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        freeze_layers: int | None = None,
        device: str | None = None,
    ):
        """Initialize MERT backbone.

        Args:
            model_name: HuggingFace model identifier. Options:
                - "m-a-p/MERT-v1-95M" (smaller, faster)
                - "m-a-p/MERT-v1-330M" (larger, more accurate)
            freeze_layers: Number of bottom layers to freeze. If None, freeze all but last 2.
            device: Torch device. If None, auto-detect CUDA.
        """
        super().__init__()

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model (MERT requires trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size

        # Freeze layers
        total_layers = len(self.model.encoder.layers)
        if freeze_layers is None:
            # Default: freeze all but last 2 layers
            freeze_layers = total_layers - 2

        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze bottom N encoder layers.

        Args:
            num_layers: Number of layers to freeze from bottom
        """
        if num_layers <= 0:
            return

        total_layers = len(self.model.encoder.layers)
        num_layers = min(num_layers, total_layers)

        # Freeze feature extractor (MERT uses feature_extractor instead of embeddings)
        if hasattr(self.model, "feature_extractor"):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False

        # Freeze encoder layers
        for i in range(num_layers):
            for param in self.model.encoder.layers[i].parameters():
                param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract frame-level embeddings from audio.

        Args:
            audio: Audio tensor [batch, samples]

        Returns:
            Frame-level embeddings [batch, time, embedding_dim]
        """
        # Ensure audio is on correct device
        audio = audio.to(self.device)

        # MERT expects [batch, samples] directly (no channel dimension)
        # Forward through model
        outputs = self.model(audio, output_hidden_states=True)

        # Return last hidden state: [batch, time, embedding_dim]
        last_hidden: torch.Tensor = outputs.last_hidden_state
        return last_hidden

    def get_embedding_dim(self) -> int:
        """Get dimensionality of output embeddings.

        Returns:
            Embedding dimension (e.g., 768 for MERT-v1-95M)
        """
        embedding_dim: int = self.embedding_dim
        return embedding_dim

    def save(self, path: Path) -> None:
        """Save backbone weights.

        Args:
            path: Path to save checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "MERTBackbone":
        """Load backbone from checkpoint.

        Args:
            path: Path to checkpoint
            device: Torch device

        Returns:
            Loaded MERTBackbone
        """
        checkpoint = torch.load(path, map_location=device or "cpu")

        # Create model
        backbone = cls(
            model_name=checkpoint["model_name"],
            device=device,
        )

        # Load weights
        backbone.model.load_state_dict(checkpoint["model_state_dict"])

        return backbone


class SimpleCNNBackbone(nn.Module):
    """Simple CNN backbone for fast training/debugging.

    This is a lightweight alternative to MERT for quick experiments.
    Uses mel spectrograms as input instead of raw audio.
    """

    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 256,
        device: str | None = None,
    ):
        """Initialize simple CNN backbone.

        Args:
            n_mels: Number of mel frequency bins
            embedding_dim: Output embedding dimension
            device: Torch device
        """
        super().__init__()

        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Conv layers on mel spectrogram
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(128, embedding_dim, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(embedding_dim)

        # Global average pooling over frequency
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))

        self.to(self.device)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Extract frame-level embeddings from mel spectrogram.

        Args:
            mel_spec: Mel spectrogram [batch, n_mels, time] or [batch, 1, n_mels, time]

        Returns:
            Frame-level embeddings [batch, time, embedding_dim]
        """
        mel_spec = mel_spec.to(self.device)

        # Add channel dimension if needed
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, n_mels, time]

        # Conv blocks
        x = torch.relu(self.bn1(self.conv1(mel_spec)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))

        # Global pool over frequency, keep time
        x = self.global_pool(x)  # [batch, embedding_dim, 1, time]
        x = x.squeeze(2)  # [batch, embedding_dim, time]
        output: torch.Tensor = x.transpose(1, 2)  # [batch, time, embedding_dim]

        return output

    def get_embedding_dim(self) -> int:
        """Get dimensionality of output embeddings."""
        return self.embedding_dim

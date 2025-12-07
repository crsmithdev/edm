"""Multi-task model combining backbone and prediction heads."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from edm.models.backbone import MERTBackbone, SimpleCNNBackbone
from edm.models.heads import BeatHead, BoundaryHead, EnergyHead, LabelHead


class MultiTaskModel(nn.Module):
    """Multi-task model for structure boundary detection, energy prediction, and beat tracking.

    Combines a shared backbone (MERT or CNN) with task-specific prediction heads.
    Supports selective training of individual heads based on priorities.
    """

    def __init__(
        self,
        backbone: nn.Module,
        enable_boundary: bool = True,
        enable_energy: bool = True,
        enable_beat: bool = True,
        enable_label: bool = False,
        num_classes: int = 6,
    ) -> None:
        """Initialize multi-task model.

        Args:
            backbone: Backbone model (MERT or SimpleCNN)
            enable_boundary: Enable boundary detection head
            enable_energy: Enable energy prediction head
            enable_beat: Enable beat detection head
            enable_label: Enable section label classification head
            num_classes: Number of label classes (auto-detected from dataset)
        """
        super().__init__()

        self.backbone = backbone
        embedding_dim: Any = backbone.get_embedding_dim()  # type: ignore[operator]
        assert isinstance(embedding_dim, int)

        # Create enabled heads
        self.enable_boundary = enable_boundary
        self.enable_energy = enable_energy
        self.enable_beat = enable_beat
        self.enable_label = enable_label

        if enable_boundary:
            self.boundary_head = BoundaryHead(input_dim=embedding_dim)
        if enable_energy:
            self.energy_head = EnergyHead(input_dim=embedding_dim, num_bands=3)
        if enable_beat:
            self.beat_head = BeatHead(input_dim=embedding_dim)
        if enable_label:
            self.label_head = LabelHead(input_dim=embedding_dim, num_classes=num_classes)

    def forward(
        self,
        audio: torch.Tensor,
    ) -> dict[str, Any]:
        """Forward pass through all enabled heads.

        Args:
            audio: Audio input (shape depends on backbone type)
                - MERT: [batch, samples] or [batch, 1, samples]
                - CNN: [batch, n_mels, time]

        Returns:
            Dict with predictions from enabled heads:
                - boundary: [batch, time, 1] if enabled
                - energy: [batch, time, 3] if enabled
                - beat: [batch, time, 1] if enabled
                - label: [batch, time, 6] if enabled
        """
        # Extract embeddings
        embeddings = self.backbone(audio)  # [batch, time, embedding_dim]

        # Run enabled heads
        outputs = {}

        if self.enable_boundary:
            outputs["boundary"] = self.boundary_head(embeddings)
        if self.enable_energy:
            outputs["energy"] = self.energy_head(embeddings)
        if self.enable_beat:
            outputs["beat"] = self.beat_head(embeddings)
        if self.enable_label:
            outputs["label"] = self.label_head(embeddings)

        return outputs

    def save(self, path: Path) -> None:
        """Save full model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "enable_boundary": self.enable_boundary,
            "enable_energy": self.enable_energy,
            "enable_beat": self.enable_beat,
            "enable_label": self.enable_label,
        }

        # Save backbone type and config
        if isinstance(self.backbone, MERTBackbone):
            checkpoint["backbone_type"] = "mert"
            checkpoint["backbone_config"] = {
                "model_name": self.backbone.model_name,
                "embedding_dim": self.backbone.embedding_dim,
            }
        elif isinstance(self.backbone, SimpleCNNBackbone):
            checkpoint["backbone_type"] = "cnn"
            checkpoint["backbone_config"] = {
                "n_mels": self.backbone.n_mels,
                "embedding_dim": self.backbone.embedding_dim,
            }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "MultiTaskModel":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Torch device

        Returns:
            Loaded MultiTaskModel
        """
        checkpoint = torch.load(path, map_location=device or "cpu")

        # Reconstruct backbone
        backbone_type = checkpoint["backbone_type"]
        backbone_config = checkpoint["backbone_config"]

        backbone: MERTBackbone | SimpleCNNBackbone
        if backbone_type == "mert":
            backbone = MERTBackbone(
                model_name=backbone_config["model_name"],
                device=device,
            )
        elif backbone_type == "cnn":
            backbone = SimpleCNNBackbone(
                n_mels=backbone_config["n_mels"],
                embedding_dim=backbone_config["embedding_dim"],
                device=device,
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Create model
        model = cls(
            backbone=backbone,
            enable_boundary=checkpoint["enable_boundary"],
            enable_energy=checkpoint["enable_energy"],
            enable_beat=checkpoint["enable_beat"],
            enable_label=checkpoint["enable_label"],
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        if device:
            model.to(device)

        return model

    def get_device(self) -> torch.device:
        """Get device the model is on."""
        return next(self.parameters()).device

    def count_parameters(self) -> dict[str, int]:
        """Count parameters in each component.

        Returns:
            Dict with parameter counts
        """
        counts = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "backbone_trainable": sum(
                p.numel() for p in self.backbone.parameters() if p.requires_grad
            ),
        }

        if self.enable_boundary:
            counts["boundary_head"] = sum(p.numel() for p in self.boundary_head.parameters())
        if self.enable_energy:
            counts["energy_head"] = sum(p.numel() for p in self.energy_head.parameters())
        if self.enable_beat:
            counts["beat_head"] = sum(p.numel() for p in self.beat_head.parameters())
        if self.enable_label:
            counts["label_head"] = sum(p.numel() for p in self.label_head.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return counts


def create_model(
    backbone_type: Literal["mert-95m", "mert-330m", "cnn"] = "mert-95m",
    enable_boundary: bool = True,
    enable_energy: bool = True,
    enable_beat: bool = True,
    enable_label: bool = False,
    num_classes: int = 6,
    device: str | None = None,
) -> MultiTaskModel:
    """Factory function to create a multi-task model.

    Args:
        backbone_type: Type of backbone to use
        enable_boundary: Enable boundary detection
        enable_energy: Enable energy prediction
        enable_beat: Enable beat detection
        enable_label: Enable label classification
        num_classes: Number of label classes (auto-detected from dataset)
        device: Torch device

    Returns:
        Configured MultiTaskModel
    """
    # Create backbone
    backbone: MERTBackbone | SimpleCNNBackbone
    if backbone_type == "mert-95m":
        backbone = MERTBackbone(
            model_name="m-a-p/MERT-v1-95M",
            device=device,
        )
    elif backbone_type == "mert-330m":
        backbone = MERTBackbone(
            model_name="m-a-p/MERT-v1-330M",
            device=device,
        )
    elif backbone_type == "cnn":
        backbone = SimpleCNNBackbone(
            n_mels=128,
            embedding_dim=256,
            device=device,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    # Create model
    model = MultiTaskModel(
        backbone=backbone,
        enable_boundary=enable_boundary,
        enable_energy=enable_energy,
        enable_beat=enable_beat,
        enable_label=enable_label,
        num_classes=num_classes,
    )

    return model

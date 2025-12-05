"""Loss functions for multi-task EDM structure detection."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining boundary, energy, beat, and label losses.

    Attributes:
        boundary_weight: Weight for boundary detection loss
        energy_weight: Weight for energy prediction loss
        beat_weight: Weight for beat detection loss
        label_weight: Weight for label classification loss
        use_focal: Use focal loss for boundary/beat (helps with class imbalance)
    """

    def __init__(
        self,
        boundary_weight: float = 1.0,
        energy_weight: float = 0.5,
        beat_weight: float = 0.5,
        label_weight: float = 0.3,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ):
        """Initialize multi-task loss.

        Args:
            boundary_weight: Weight for boundary detection
            energy_weight: Weight for energy prediction
            beat_weight: Weight for beat detection
            label_weight: Weight for label classification
            use_focal: Use focal loss for imbalanced tasks
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()

        self.boundary_weight = boundary_weight
        self.energy_weight = energy_weight
        self.beat_weight = beat_weight
        self.label_weight = label_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            predictions: Dict with model predictions
                - boundary: [batch, time, 1]
                - energy: [batch, time, 3]
                - beat: [batch, time, 1]
                - label: [batch, time, num_classes]
            targets: Dict with ground truth targets
                - boundary: [batch, time, 1]
                - energy: [batch, time, 3]
                - beat: [batch, time, 1]
                - label: [batch, time]

        Returns:
            Dict with losses:
                - total: Combined weighted loss
                - boundary: Boundary loss
                - energy: Energy loss
                - beat: Beat loss
                - label: Label loss (if present)
        """
        losses = {}

        # Boundary detection loss (binary classification)
        if "boundary" in predictions:
            if self.use_focal:
                losses["boundary"] = self._focal_binary_loss(
                    predictions["boundary"],
                    targets["boundary"],
                )
            else:
                losses["boundary"] = functional.binary_cross_entropy(
                    predictions["boundary"],
                    targets["boundary"],
                    reduction="mean",
                )

        # Energy prediction loss (regression)
        if "energy" in predictions:
            losses["energy"] = functional.mse_loss(
                predictions["energy"],
                targets["energy"],
                reduction="mean",
            )

        # Beat detection loss (binary classification)
        if "beat" in predictions:
            if self.use_focal:
                losses["beat"] = self._focal_binary_loss(
                    predictions["beat"],
                    targets["beat"],
                )
            else:
                losses["beat"] = functional.binary_cross_entropy(
                    predictions["beat"],
                    targets["beat"],
                    reduction="mean",
                )

        # Label classification loss (multi-class)
        if "label" in predictions:
            # Reshape for cross entropy: [batch * time, num_classes]
            pred = predictions["label"].reshape(-1, predictions["label"].shape[-1])
            target = targets["label"].reshape(-1)

            losses["label"] = functional.cross_entropy(
                pred,
                target,
                reduction="mean",
            )

        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        if "boundary" in losses:
            total_loss += self.boundary_weight * losses["boundary"]
        if "energy" in losses:
            total_loss += self.energy_weight * losses["energy"]
        if "beat" in losses:
            total_loss += self.beat_weight * losses["beat"]
        if "label" in losses:
            total_loss += self.label_weight * losses["label"]

        losses["total"] = total_loss

        return losses

    def _focal_binary_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss for binary classification.

        Helps with class imbalance by down-weighting easy examples.

        Args:
            pred: Predictions [batch, time, 1] in range [0, 1]
            target: Targets [batch, time, 1]

        Returns:
            Focal loss scalar
        """
        # Compute binary cross entropy
        bce = functional.binary_cross_entropy(pred, target, reduction="none")

        # Compute focal weight
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Apply focal weight
        focal_loss = focal_weight * bce

        return focal_loss.mean()


class BoundaryF1Loss(nn.Module):
    """F1-optimized loss for boundary detection.

    Directly optimizes for F1 score by using a differentiable approximation.
    """

    def __init__(self, beta: float = 1.0, epsilon: float = 1e-7):
        """Initialize F1 loss.

        Args:
            beta: F-beta score parameter (1.0 = F1)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute F1 loss.

        Args:
            pred: Predictions [batch, time, 1]
            target: Targets [batch, time, 1]

        Returns:
            F1 loss (1 - F1 score)
        """
        # Compute true positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        # Compute precision and recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        # Compute F-beta score
        beta_squared = self.beta**2
        f_beta = (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall + self.epsilon)
        )

        # Return 1 - F_beta as loss (to minimize)
        return 1 - f_beta


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for energy prediction.

    Gives higher weight to high-energy regions (drops).
    """

    def __init__(self, high_energy_weight: float = 2.0):
        """Initialize weighted MSE loss.

        Args:
            high_energy_weight: Weight multiplier for high-energy frames
        """
        super().__init__()
        self.high_energy_weight = high_energy_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            pred: Predictions [batch, time, 3]
            target: Targets [batch, time, 3]

        Returns:
            Weighted MSE loss
        """
        # Compute per-frame energy (mean across bands)
        target_energy = target.mean(dim=-1, keepdim=True)  # [batch, time, 1]

        # Compute weights based on energy level
        # High energy (>0.7) gets higher weight
        weights = torch.where(
            target_energy > 0.7,
            torch.ones_like(target_energy) * self.high_energy_weight,
            torch.ones_like(target_energy),
        )

        # Compute weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = mse * weights

        return weighted_mse.mean()

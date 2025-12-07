"""Task-specific prediction heads for multi-task learning."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class BoundaryHead(nn.Module):
    """Frame-wise boundary detection head.

    Predicts probability of a structure boundary at each time frame.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        kernel_size: int = 7,
    ):
        """Initialize boundary head.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            kernel_size: Temporal convolution kernel size
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.out = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict boundary probabilities.

        Args:
            x: Input embeddings [batch, time, feat]

        Returns:
            Boundary probabilities [batch, time, 1]
        """
        # Transpose for conv1d: [batch, feat, time]
        x = x.transpose(1, 2)

        # Conv blocks
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = functional.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Output layer
        x = torch.sigmoid(self.out(x))

        # Transpose back: [batch, time, 1]
        return x.transpose(1, 2)


class EnergyHead(nn.Module):
    """Frame-wise energy regression head.

    Predicts multi-band energy levels (bass, mid, high) at each time frame.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_bands: int = 3,
        kernel_size: int = 7,
    ):
        """Initialize energy head.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_bands: Number of frequency bands (default: 3 for bass/mid/high)
            kernel_size: Temporal convolution kernel size
        """
        super().__init__()

        padding = kernel_size // 2
        self.num_bands = num_bands

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.out = nn.Conv1d(hidden_dim, num_bands, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict energy levels.

        Args:
            x: Input embeddings [batch, time, feat]

        Returns:
            Energy levels [batch, time, num_bands] in range [0, 1]
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)

        # Conv blocks
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = functional.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Output layer with sigmoid for [0, 1] range
        x = torch.sigmoid(self.out(x))

        # Transpose back
        return x.transpose(1, 2)


class BeatHead(nn.Module):
    """Frame-wise beat detection head.

    Predicts probability of a beat at each time frame.
    Used to improve beat tracking beyond DJ software metadata.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        kernel_size: int = 7,
    ):
        """Initialize beat head.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            kernel_size: Temporal convolution kernel size
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim // 2, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.out = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict beat probabilities.

        Args:
            x: Input embeddings [batch, time, feat]

        Returns:
            Beat probabilities [batch, time, 1]
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)

        # Conv blocks
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = functional.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Output layer
        x = torch.sigmoid(self.out(x))

        # Transpose back
        return x.transpose(1, 2)


class LabelHead(nn.Module):
    """Frame-wise section label classification head.

    Predicts section label (intro, buildup, drop, breakdown, outro) at each frame.
    This is optional - priority 4 in our implementation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 6,
        kernel_size: int = 11,
    ):
        """Initialize label head.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of section classes (default: 6 for intro/buildup/drop/breakdown/outro/unlabeled)
            kernel_size: Temporal convolution kernel size (larger for label context)
        """
        super().__init__()

        padding = kernel_size // 2
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.out = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict section labels.

        Args:
            x: Input embeddings [batch, time, feat]

        Returns:
            Label logits [batch, time, num_classes]
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)

        # Conv blocks
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = functional.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Output layer (logits, no activation)
        x = self.out(x)

        # Transpose back
        return x.transpose(1, 2)

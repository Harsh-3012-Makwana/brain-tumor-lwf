"""Full Custom CNN architecture integrating 4-block backbone and expandable head."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.custom_cnn.backbone import CustomCNNBackbone
from src.models.heads.expandable_head import ExpandableLinearHead


class CustomCNN(nn.Module):
    """Custom Lightweight CNN (<5M parameters) for Class-Incremental MRI Classification.

    Composed of:
        1. 4-block ConvNet backbone with BatchNorm, ReLU, MaxPool.
        2. Global Average Pooling (GAP) reducing spatial feature maps to (B, C).
        3. Dropout(0.4) for regularization.
        4. Expandable linear classification head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Optional[List[int]] = None,
        convs_per_block: int = 1,
        dropout_rates: Optional[List[float]] = None,
        gap_dropout: float = 0.4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = CustomCNNBackbone(
            in_channels=in_channels,
            stage_channels=stage_channels,
            convs_per_block=convs_per_block,
            dropout_rates=dropout_rates,
            gap_dropout=gap_dropout,
        )
        self.head = ExpandableLinearHead(
            in_features=self.backbone.out_features,
            initial_num_classes=num_classes,
        )

    @property
    def num_classes(self) -> int:
        """Current number of active output classes."""
        return self.head.num_classes

    @property
    def feature_dim(self) -> int:
        """Dimensionality of pooled feature representations."""
        return self.backbone.out_features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 1D feature representation (B, feature_dim).

        Args:
            x: Input image tensor of shape (B, 3, 150, 150).
        """
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits across all currently active classes.

        Args:
            x: Input image tensor of shape (B, 3, 150, 150).

        Returns:
            Logits of shape (B, num_classes) in global label order.
        """
        features = self.extract_features(x)
        return self.head.forward(features)

    def forward_old_logits(self, x: torch.Tensor, num_old_classes: int) -> torch.Tensor:
        """Compute logits restricted strictly to the previous old classes.

        Facilitates LwF distillation loss computation without requiring extra head slicing in training loops.

        Args:
            x: Input image tensor of shape (B, 3, 150, 150).
            num_old_classes: Count of previous classes.

        Returns:
            Logits of shape (B, num_old_classes).
        """
        features = self.extract_features(x)
        return self.head.forward_old(features, num_old_classes)

    @staticmethod
    def get_old_logits(logits: torch.Tensor, num_old_classes: int) -> torch.Tensor:
        """Convenience utility to slice old-class logits from a full prediction tensor."""
        if num_old_classes > logits.shape[1]:
            raise ValueError(
                f"Requested {num_old_classes} old classes from tensor with only {logits.shape[1]} outputs"
            )
        return logits[:, :num_old_classes]

    def expand(self, n_new_classes: int) -> None:
        """Expand the classification head to support novel classes arriving in an incremental step.

        Preserves existing class weights bit-identically while appending new output neurons.
        """
        self.head.expand(n_new_classes)

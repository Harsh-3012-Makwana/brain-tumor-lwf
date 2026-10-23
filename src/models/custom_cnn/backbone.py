"""Configurable 4-block CNN backbone with Global Average Pooling."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block: Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2.

    Optionally repeats Conv3x3 -> BatchNorm -> ReLU if convs_per_block > 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs_per_block: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        curr_in = in_channels
        for i in range(convs_per_block):
            layers.append(
                nn.Conv2d(
                    curr_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,  # Bias is redundant before BatchNorm
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            curr_in = out_channels

        layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNNBackbone(nn.Module):
    """4-block convolutional feature extractor with Global Average Pooling (GAP).

    Exposes forward_features to produce pooled feature embeddings separate from classification heads,
    and forward_conv to retain 4D feature activations (for Grad-CAM).
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Optional[List[int]] = None,
        convs_per_block: int = 1,
        dropout_rates: Optional[List[float]] = None,
        gap_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if stage_channels is None:
            stage_channels = [32, 64, 128, 128]
        if len(stage_channels) != 4:
            raise ValueError(f"Expected exactly 4 stages, got {len(stage_channels)}")

        if dropout_rates is None:
            dropout_rates = [0.0, 0.0, 0.0, 0.0]

        self.stage_channels = stage_channels
        self.convs_per_block = convs_per_block
        self.out_features = stage_channels[-1]

        # 4 Convolutional stages
        blocks: List[nn.Module] = []
        curr_in = in_channels
        for out_c, drop in zip(stage_channels, dropout_rates):
            blocks.append(
                ConvBlock(
                    in_channels=curr_in,
                    out_channels=out_c,
                    convs_per_block=convs_per_block,
                    dropout_rate=drop,
                )
            )
            curr_in = out_c

        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=gap_dropout)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial 4D feature map before pooling (B, C, H, W)."""
        return self.blocks(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled and regularized 1D feature vector (B, out_features)."""
        feat_map = self.forward_conv(x)
        pooled = self.global_pool(feat_map)
        flat = self.flatten(pooled)
        return self.dropout(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass through backbone returns pooled 1D features."""
        return self.forward_features(x)

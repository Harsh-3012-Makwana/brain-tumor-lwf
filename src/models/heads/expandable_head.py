"""Expandable linear classification head for class-incremental continual learning."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ExpandableLinearHead(nn.Module):
    """Linear classification head supporting dynamic expansion for class-incremental learning.

    Preserves previously learned output weights and biases bit-identically while
    initializing newly appended class weights.
    """

    def __init__(self, in_features: int, initial_num_classes: int = 2) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if initial_num_classes <= 0:
            raise ValueError(f"initial_num_classes must be positive, got {initial_num_classes}")

        self.in_features = in_features
        self.num_classes = initial_num_classes

        self.linear = nn.Linear(in_features, initial_num_classes, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using standard Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def expand(self, n_new_classes: int) -> None:
        """Dynamically expand head to accommodate additional novel classes.

        Args:
            n_new_classes: Count of novel classes to append to the output layer.
        """
        if n_new_classes <= 0:
            raise ValueError(f"n_new_classes must be strictly positive, got {n_new_classes}")

        old_num_classes = self.num_classes
        new_total_classes = old_num_classes + n_new_classes

        # Instantiate expanded linear layer matching device and dtype
        new_linear = nn.Linear(
            self.in_features,
            new_total_classes,
            bias=True,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )

        with torch.no_grad():
            # 1. Exact bit-identical copy of previous class weights and biases
            new_linear.weight[:old_num_classes].copy_(self.linear.weight)
            new_linear.bias[:old_num_classes].copy_(self.linear.bias)

            # 2. Sensible random initialization for new classes
            nn.init.kaiming_uniform_(new_linear.weight[old_num_classes:], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_linear.weight[old_num_classes:])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(new_linear.bias[old_num_classes:], -bound, bound)

        self.linear = new_linear
        self.num_classes = new_total_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits across all current active classes.

        Args:
            features: Pooled 1D feature representations of shape (B, in_features).

        Returns:
            Logit tensor of shape (B, num_classes) in global label order.
        """
        return self.linear(features)

    def forward_old(self, features: torch.Tensor, n_old_classes: int) -> torch.Tensor:
        """Compute logits restricted strictly to the previous old classes.

        Args:
            features: Pooled 1D feature representations of shape (B, in_features).
            n_old_classes: Count of previous classes.

        Returns:
            Logit tensor of shape (B, n_old_classes).
        """
        if n_old_classes > self.num_classes:
            raise ValueError(
                f"Requested {n_old_classes} old classes, but head only has {self.num_classes}"
            )
        full_logits = self.forward(features)
        return full_logits[:, :n_old_classes]

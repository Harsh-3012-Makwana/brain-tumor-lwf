"""Teacher model snapshot management for Learning without Forgetting (LwF)."""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_frozen_teacher(model: nn.Module) -> nn.Module:
    """Create an immutable, frozen deep-copy of the model to serve as distillation teacher.

    Ensures the teacher remains in eval() mode with all gradient computations disabled.

    Args:
        model: Trained PyTorch model to snapshot.

    Returns:
        Frozen copy of the model in eval mode.
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    logger.debug(
        "Created frozen teacher snapshot with %d parameters and %d active output classes",
        sum(p.numel() for p in teacher.parameters()),
        getattr(teacher, "num_classes", 0),
    )
    return teacher

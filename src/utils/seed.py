"""Reproducibility utilities for seed management across Python, NumPy, and PyTorch.

Ensures deterministic behavior across data loaders, weight initializations,
and training runs.
"""

from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seed across Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: Integer seed value to set. Default is 42.
        deterministic: If True, configures PyTorch and cuDNN to use deterministic
            algorithms and disables benchmark mode.

    Returns:
        The integer seed that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # warn_only avoids crashes if specific non-deterministic CUDA ops are invoked
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception as exc:  # pragma: no cover
                    logger.debug("Failed setting use_deterministic_algorithms: %s", exc)

    except ImportError:  # pragma: no cover
        logger.debug("PyTorch is not available; set seed for random and numpy only.")

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialization function ensuring distinct, deterministic seeds per worker."""
    worker_seed = (torch_initial_seed := _get_torch_initial_seed()) + worker_id
    np.random.seed(worker_seed % (2**32))
    random.seed(worker_seed % (2**32))


def _get_torch_initial_seed() -> int:
    """Helper to retrieve PyTorch initial seed safely."""
    try:
        import torch

        return torch.initial_seed() % (2**32)
    except (ImportError, AttributeError):
        return 42

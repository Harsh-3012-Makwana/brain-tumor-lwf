"""Package module for continual learning metrics."""

from src.evaluation.incremental_metrics.continual import (
    compute_continual_metrics,
    compute_restricted_old_class_metrics,
)

__all__ = ["compute_continual_metrics", "compute_restricted_old_class_metrics"]

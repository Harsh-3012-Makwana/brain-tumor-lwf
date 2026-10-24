"""Replay DataLoader wrapper interleaving new task batches with exemplar buffer samples."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.incremental.buffers.class_balanced_buffer import ClassBalancedBuffer


class ReplayDataLoader:
    """Wraps a task DataLoader and appends a buffer batch drawn with replacement."""

    def __init__(
        self,
        base_loader: DataLoader,
        buffer: Optional[ClassBalancedBuffer] = None,
        max_buffer_batch_size: int = 32,
    ) -> None:
        """Initialize ReplayDataLoader.

        Args:
            base_loader: Underlying DataLoader providing batches of new-task data.
            buffer: Active ClassBalancedBuffer (or None if no replay data available).
            max_buffer_batch_size: Maximum buffer exemplars drawn per step (default 32).
        """
        self.base_loader = base_loader
        self.buffer = buffer
        self.max_buffer_batch_size = max_buffer_batch_size

    def __len__(self) -> int:
        return len(self.base_loader)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Any]]:
        for batch in self.base_loader:
            images_new = batch[0]
            targets_new = batch[1]
            paths_new = batch[2] if len(batch) > 2 else []

            if self.buffer is not None and self.buffer.total_samples > 0:
                buf_batch_size = min(self.max_buffer_batch_size, self.buffer.total_samples)
                images_buf, targets_buf = self.buffer.sample(buf_batch_size)

                # Concatenate along batch dimension
                images_union = torch.cat([images_new, images_buf.to(images_new.device)], dim=0)
                targets_union = torch.cat([targets_new, targets_buf.to(targets_new.device)], dim=0)
                paths_union = (
                    paths_new + ["buffer_replay"] * len(targets_buf)
                    if isinstance(paths_new, list)
                    else paths_new
                )

                yield images_union, targets_union, paths_union
            else:
                yield images_new, targets_new, paths_new

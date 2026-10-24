"""Class-balanced experience replay buffer for continual learning."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ClassBalancedBuffer:
    """Stores a fixed total capacity M of raw image exemplars balanced across seen classes.

    Ensures zero leakage across partitions by exclusively sampling from the
    training split of completed tasks.
    """

    def __init__(self, capacity: int, seed: int = 42) -> None:
        """Initialize ClassBalancedBuffer.

        Args:
            capacity: Total exemplar budget M across all seen classes (e.g. 60 or 120).
            seed: Random seed for deterministic sample selection and replacement sampling.
        """
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.seed = seed
        self.rng = random.Random(seed)

        # Storage organized by global class id: {class_id: [(image_tensor, global_label, path), ...]}
        self.storage: Dict[int, List[Tuple[torch.Tensor, int, str]]] = {}

    @property
    def total_samples(self) -> int:
        """Current total number of stored exemplars."""
        return sum(len(items) for items in self.storage.values())

    @property
    def seen_classes(self) -> List[int]:
        """List of global class IDs currently present in buffer."""
        return sorted(list(self.storage.keys()))

    def update_buffer(self, seen_classes: List[int], task_train_dataset: Dataset) -> None:
        """Update buffer after completing a task to include exemplars of novel classes.

        Rebalances the buffer so that capacity M is divided evenly across all
        currently seen classes.

        Args:
            seen_classes: All global class IDs seen up to and including the finished task.
            task_train_dataset: The training dataset of the newly completed task.
        """
        k = len(seen_classes)
        if k == 0:
            return

        target_per_class = self.capacity // k

        # 1. Downsample existing stored classes to target_per_class
        for cls_id in list(self.storage.keys()):
            if len(self.storage[cls_id]) > target_per_class:
                self.rng.shuffle(self.storage[cls_id])
                self.storage[cls_id] = self.storage[cls_id][:target_per_class]

        # 2. Extract and sample novel classes from the finished task's training dataset
        # Group task dataset items by global label
        new_samples_by_class: Dict[int, List[Tuple[torch.Tensor, int, str]]] = {}
        for idx in range(len(task_train_dataset)):
            img, label, path = task_train_dataset[idx]
            if label not in new_samples_by_class:
                new_samples_by_class[label] = []
            new_samples_by_class[label].append((img, label, path))

        for cls_id in seen_classes:
            if cls_id not in self.storage:
                candidates = new_samples_by_class.get(cls_id, [])
                if len(candidates) > target_per_class:
                    self.rng.shuffle(candidates)
                    self.storage[cls_id] = candidates[:target_per_class]
                else:
                    self.storage[cls_id] = list(candidates)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of exemplars uniformly at random with replacement.

        Args:
            batch_size: Number of exemplars to sample.

        Returns:
            Tuple of (images_tensor, targets_tensor).
        """
        all_items: List[Tuple[torch.Tensor, int, str]] = []
        for items in self.storage.values():
            all_items.extend(items)

        if not all_items:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        sampled = [self.rng.choice(all_items) for _ in range(batch_size)]
        images = torch.stack([item[0] for item in sampled], dim=0)
        targets = torch.tensor([item[1] for item in sampled], dtype=torch.long)

        return images, targets

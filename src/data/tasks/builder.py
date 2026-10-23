"""Class-incremental task builder and experience replay buffer sampler."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Subset

from src.data.augmentation.augment import get_eval_transforms, get_train_transforms
from src.data.datasets.mri_dataset import MRIDataset
from src.utils.paths import SPLITS_DIR

logger = logging.getLogger(__name__)

# Frozen task orders
TASK_ORDERS: Dict[str, List[List[int]]] = {
    "A": [
        [0, 1],  # Task 1: glioma (0), meningioma (1)
        [3],     # Task 2: pituitary (3)
        [2],     # Task 3: notumor (2)
    ],
    "B": [
        [2, 3],  # Task 1: notumor (2), pituitary (3)
        [0],     # Task 2: glioma (0)
        [1],     # Task 3: meningioma (1)
    ],
}


@dataclass
class TaskData:
    """Encapsulates dataset partitions and metadata for a specific incremental step."""

    task_id: int
    order: str
    seed: int
    new_classes: List[int]
    seen_classes: List[int]
    head_size: int
    train_dataset: MRIDataset
    val_dataset: MRIDataset
    cumulative_val_dataset: MRIDataset
    cumulative_test_dataset: MRIDataset


def get_task_datasets(
    order: str,
    seed: int,
    image_size: Tuple[int, int] = (150, 150),
    splits_dir: Path = SPLITS_DIR,
) -> List[TaskData]:
    """Construct sequential task datasets for a given class order and random seed.

    Args:
        order: Task order sequence ('A' or 'B').
        seed: Random seed (42 or 123).
        image_size: Target image spatial dimension.
        splits_dir: Directory containing split CSVs.

    Returns:
        List of 3 TaskData instances (one for Task 1, Task 2, Task 3).
    """
    order = order.upper()
    if order not in TASK_ORDERS:
        raise ValueError(f"Unknown task order: {order}. Supported orders: ['A', 'B']")

    train_val_csv = splits_dir / f"train_val_seed{seed}.csv"
    test_csv = splits_dir / "test.csv"

    if not train_val_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found at {train_val_csv}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV not found at {test_csv}")

    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)

    task_stages = TASK_ORDERS[order]
    seen_classes: List[int] = []
    tasks: List[TaskData] = []

    for task_idx, new_classes in enumerate(task_stages, start=1):
        seen_classes = seen_classes + new_classes
        head_size = len(seen_classes)

        # 1. Training dataset: only new classes for this task
        train_ds = MRIDataset(
            split_csv_path=train_val_csv,
            split="train",
            allowed_classes=new_classes,
            transform=train_transform,
        )

        # 2. Validation dataset for task-specific new classes
        val_ds = MRIDataset(
            split_csv_path=train_val_csv,
            split="val",
            allowed_classes=new_classes,
            transform=eval_transform,
        )

        # 3. Cumulative validation dataset across all seen classes (for checkpoint selection)
        cum_val_ds = MRIDataset(
            split_csv_path=train_val_csv,
            split="val",
            allowed_classes=seen_classes,
            transform=eval_transform,
        )

        # 4. Cumulative test dataset across all seen classes
        cum_test_ds = MRIDataset(
            split_csv_path=test_csv,
            split="test",
            allowed_classes=seen_classes,
            transform=eval_transform,
        )

        task_data = TaskData(
            task_id=task_idx,
            order=order,
            seed=seed,
            new_classes=list(new_classes),
            seen_classes=list(seen_classes),
            head_size=head_size,
            train_dataset=train_ds,
            val_dataset=val_ds,
            cumulative_val_dataset=cum_val_ds,
            cumulative_test_dataset=cum_test_ds,
        )
        tasks.append(task_data)

    return tasks


def get_joint_datasets(
    seed: int,
    image_size: Tuple[int, int] = (150, 150),
    splits_dir: Path = SPLITS_DIR,
) -> Tuple[MRIDataset, MRIDataset, MRIDataset]:
    """Return all 4 classes at once for Joint Training (upper bound).

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_val_csv = splits_dir / f"train_val_seed{seed}.csv"
    test_csv = splits_dir / "test.csv"

    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)

    train_ds = MRIDataset(
        split_csv_path=train_val_csv,
        split="train",
        allowed_classes=[0, 1, 2, 3],
        transform=train_transform,
    )
    val_ds = MRIDataset(
        split_csv_path=train_val_csv,
        split="val",
        allowed_classes=[0, 1, 2, 3],
        transform=eval_transform,
    )
    test_ds = MRIDataset(
        split_csv_path=test_csv,
        split="test",
        allowed_classes=[0, 1, 2, 3],
        transform=eval_transform,
    )
    return train_ds, val_ds, test_ds


def sample_replay_buffer(
    dataset: MRIDataset,
    buffer_size: int,
    seed: int = 42,
) -> Subset:
    """Sample a class-balanced random subset of images from finished tasks.

    Args:
        dataset: Source MRIDataset containing images from old classes.
        buffer_size: Total replay memory capacity (e.g. 60 or 120).
        seed: Random seed for deterministic subset selection.

    Returns:
        PyTorch Subset wrapping the selected buffer samples.
    """
    if buffer_size <= 0:
        raise ValueError(f"buffer_size must be positive, got {buffer_size}")

    # Group sample indices by global label
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        by_class[sample["global_label"]].append(idx)

    classes = sorted(by_class.keys())
    num_classes = len(classes)
    if num_classes == 0:
        return Subset(dataset, [])

    base_per_class = buffer_size // num_classes
    remainder = buffer_size % num_classes

    rng = random.Random(seed)
    selected_indices: List[int] = []

    for i, cls_id in enumerate(classes):
        target_k = base_per_class + (1 if i < remainder else 0)
        cls_indices = by_class[cls_id]
        if len(cls_indices) <= target_k:
            selected_indices.extend(cls_indices)
        else:
            selected = rng.sample(cls_indices, target_k)
            selected_indices.extend(selected)

    # Sort indices for deterministic access order
    selected_indices.sort()
    return Subset(dataset, selected_indices)

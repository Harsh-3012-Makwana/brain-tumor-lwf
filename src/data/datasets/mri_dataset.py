"""PyTorch Dataset implementation for Brain Tumor MRI classification."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLOBAL_LABEL_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}


class MRIDataset(Dataset):
    """PyTorch Dataset reading standardized MRI scans from split CSV files.

    Each sample returns:
        (image_tensor, global_label, file_path_str)
    """

    def __init__(
        self,
        split_csv_path: Union[str, Path],
        split: Optional[str] = None,
        allowed_classes: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        root_dir: Optional[Path] = PROJECT_ROOT,
    ) -> None:
        """Initialize MRIDataset.

        Args:
            split_csv_path: Path to split CSV (train_val_seed42.csv, test.csv, etc.)
            split: Filter to specific split ('train', 'val', 'test', or None for all rows in CSV).
            allowed_classes: Optional list of global label integers (e.g. [0, 1] for Task 1).
            transform: Albumentations transform pipeline.
            root_dir: Base directory to resolve relative file paths.
        """
        self.csv_path = Path(split_csv_path)
        self.split = split
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else None
        self.transform = transform
        self.root_dir = root_dir or PROJECT_ROOT

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Split CSV not found at {self.csv_path}")

        self.samples: List[Dict[str, Any]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split if requested
                if self.split is not None and row["split"] != self.split:
                    continue

                label = int(row["global_label"])
                # Filter by allowed classes if incremental task specified
                if self.allowed_classes is not None and label not in self.allowed_classes:
                    continue

                self.samples.append({
                    "path": row["path"],
                    "class_name": row["class_name"],
                    "global_label": label,
                    "split": row["split"],
                })

        logger.debug(
            "Loaded MRIDataset from %s (split=%s, allowed_classes=%s): %d samples",
            self.csv_path.name,
            self.split,
            self.allowed_classes,
            len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.samples[idx]
        img_rel_path = item["path"]
        img_full_path = self.root_dir / img_rel_path

        # Read RGB image
        img_bgr = cv2.imread(str(img_full_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_full_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transform if provided
        if self.transform is not None:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"]
        else:
            # Fallback basic conversion (HWC uint8 -> CHW float [0, 1])
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Assert correct channel and spatial shape (3, 150, 150)
        assert img_tensor.shape == (3, 150, 150), (
            f"Expected image shape (3, 150, 150), got {img_tensor.shape} for {img_full_path}"
        )

        global_label = item["global_label"]
        return img_tensor, global_label, str(img_rel_path)

    def get_labels(self) -> List[int]:
        """Return list of all global labels in this dataset subset."""
        return [s["global_label"] for s in self.samples]

    def compute_class_weights(self, num_classes: int = 4) -> torch.Tensor:
        """Compute balanced inverse class frequency weights."""
        labels = self.get_labels()
        total_samples = len(labels)
        if total_samples == 0:
            return torch.ones(num_classes, dtype=torch.float32)

        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for lbl in labels:
            if 0 <= lbl < num_classes:
                class_counts[lbl] += 1.0

        weights = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            if class_counts[c] > 0:
                weights[c] = total_samples / (num_classes * class_counts[c])
            else:
                weights[c] = 0.0

        return weights

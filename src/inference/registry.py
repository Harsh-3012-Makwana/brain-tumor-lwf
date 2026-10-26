"""Model registry and checkpoint loader for MRI continual learning models.

Supports loading both 2-class (Task 1) and 4-class (Final) models from state dicts,
dynamically configuring the ExpandableLinearHead, and resolving known class mappings.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from src.models.custom_cnn.factory import build_model
from src.models.custom_cnn.model import CustomCNN
from src.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

GLOBAL_CLASSES_4 = ["glioma", "meningioma", "notumor", "pituitary"]
ORDER_C_TASK1_CLASSES = ["glioma", "meningioma"]
ORDER_D_TASK1_CLASSES = ["notumor", "pituitary"]


@dataclass
class ModelMetadata:
    """Metadata describing an evaluated model checkpoint."""

    key: str
    display_name: str
    pattern: str
    known_classes: List[str]
    variant: str = "medium"
    order: str = "Joint"
    task_description: str = "Final"
    memory_description: str = "0 images"
    info_line: str = ""

    def resolve_checkpoint_path(self, root: Optional[Path] = None) -> Optional[Path]:
        """Find the exact checkpoint file on disk matching pattern."""
        base = root or PROJECT_ROOT
        matches = sorted(glob.glob(str(base / self.pattern)))
        if not matches:
            return None
        return Path(matches[-1])


# Registry of 11 demo models (Joint, Order C, Order D; all Seed 42)
MODEL_REGISTRY_CONFIGS: List[ModelMetadata] = [
    ModelMetadata(
        key="joint_seed42",
        display_name="Joint Training (Upper Bound) - Seed 42",
        pattern="results/runs/joint_none_42_medium_*/best.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Joint",
        task_description="All 4 Classes Jointly",
        memory_description="All training data accessible (Upper Bound Oracle)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Joint Upper Bound)",
    ),
    ModelMetadata(
        key="order_c_task1_seed42",
        display_name="Order C: After Task 1 (glioma, meningioma) - Seed 42",
        pattern="results/checkpoints/shared_task1_order_c_seed42.pt",
        known_classes=ORDER_C_TASK1_CLASSES,
        order="Order C",
        task_description="Task 1 Baseline (glioma, meningioma)",
        memory_description="0 images (Task 1 initial checkpoint)",
        info_line="Architecture: Custom CNN Medium (1,035,970 params) | Known Classes: glioma, meningioma (2 classes) | Memory: 0 images (Task 1 Checkpoint)",
    ),
    ModelMetadata(
        key="order_c_finetune_seed42",
        display_name="Order C: Fine-Tuning Final - Seed 42",
        pattern="results/runs/finetune_c_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order C",
        task_description="Final (Task 2 finished)",
        memory_description="0 images (Naive Sequential Fine-Tuning lower bound)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Naive Fine-Tuning)",
    ),
    ModelMetadata(
        key="order_c_lwf_all_seed42",
        display_name="Order C: LwF (Standard / All CE) Final - Seed 42",
        pattern="results/runs/lwf_all_c_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order C",
        task_description="Final (Task 2 finished)",
        memory_description="0 images (Teacher Copy: 3.96 MB, Standard LwF with all-class CE)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Teacher: 3.96 MB, Standard LwF)",
    ),
    ModelMetadata(
        key="order_c_lwf_new_seed42",
        display_name="Order C: LwF (New-Class CE) Final - Seed 42",
        pattern="results/runs/lwf_new_c_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order C",
        task_description="Final (Task 2 finished)",
        memory_description="0 images (Teacher Copy: 3.96 MB, Modified LwF with new-class CE)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Teacher: 3.96 MB, New-Class CE)",
    ),
    ModelMetadata(
        key="order_c_replay120_seed42",
        display_name="Order C: Experience Replay (M=120) Final - Seed 42",
        pattern="results/runs/replay120_c_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order C",
        task_description="Final (Task 2 finished)",
        memory_description="120 exemplar images in replay memory (30 images per class)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 120 exemplar images (30/class)",
    ),
    ModelMetadata(
        key="order_d_task1_seed42",
        display_name="Order D: After Task 1 (notumor, pituitary) - Seed 42",
        pattern="results/checkpoints/shared_task1_order_d_seed42.pt",
        known_classes=ORDER_D_TASK1_CLASSES,
        order="Order D",
        task_description="Task 1 Baseline (notumor, pituitary)",
        memory_description="0 images (Task 1 initial checkpoint)",
        info_line="Architecture: Custom CNN Medium (1,035,970 params) | Known Classes: notumor, pituitary (2 classes) | Memory: 0 images (Task 1 Checkpoint)",
    ),
    ModelMetadata(
        key="order_d_finetune_seed42",
        display_name="Order D: Fine-Tuning Final - Seed 42",
        pattern="results/runs/finetune_d_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order D",
        task_description="Final (Task 2 finished)",
        memory_description="0 images (Naive Sequential Fine-Tuning lower bound)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Naive Fine-Tuning)",
    ),
    ModelMetadata(
        key="order_d_lwf_all_seed42",
        display_name="Order D: LwF (Standard / All CE) Final - Seed 42",
        pattern="results/runs/lwf_all_d_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order D",
        task_description="Final (Task 2 finished)",
        memory_description="0 images (Teacher Copy: 3.96 MB, Standard LwF with all-class CE)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Teacher: 3.96 MB, Standard LwF)",
    ),
    ModelMetadata(
        key="order_d_lwf_new_seed42",
        display_name="Order D: LwF (New-Class CE) Final - Seed 42",
        pattern="results/runs/lwf_new_d_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order D",
        task_description="Final (Task 2 finished)",
        memory_description="0 images (Teacher Copy: 3.96 MB, Modified LwF with new-class CE)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 0 images (Teacher: 3.96 MB, New-Class CE)",
    ),
    ModelMetadata(
        key="order_d_replay120_seed42",
        display_name="Order D: Experience Replay (M=120) Final - Seed 42",
        pattern="results/runs/replay120_d_42_medium_*/checkpoint_task2.pt",
        known_classes=GLOBAL_CLASSES_4,
        order="Order D",
        task_description="Final (Task 2 finished)",
        memory_description="120 exemplar images in replay memory (30 images per class)",
        info_line="Architecture: Custom CNN Medium (1,036,548 params) | Known Classes: glioma, meningioma, notumor, pituitary (4 classes) | Memory: 120 exemplar images (30/class)",
    ),
]

_REGISTRY_BY_NAME: Dict[str, ModelMetadata] = {
    m.display_name: m for m in MODEL_REGISTRY_CONFIGS
}

_LOADED_CACHE: Dict[str, Tuple[CustomCNN, List[str], str]] = {}


def get_available_model_names() -> List[str]:
    """Return list of display names for all models in the registry that exist on disk."""
    available: List[str] = []
    for meta in MODEL_REGISTRY_CONFIGS:
        path = meta.resolve_checkpoint_path()
        if path is not None and path.is_file():
            available.append(meta.display_name)
    return available


def get_model_metadata(display_name: str) -> Optional[ModelMetadata]:
    """Retrieve metadata for a given model display name."""
    return _REGISTRY_BY_NAME.get(display_name)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    variant: str = "medium",
) -> Tuple[CustomCNN, int]:
    """Load CustomCNN model from checkpoint, dynamically setting head dimension.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to place model on ('cuda', 'cpu', etc.).
        variant: Width variant ('small', 'medium', 'large').

    Returns:
        Tuple of (model_in_eval_mode, head_size).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")

    # Inspect head dimension
    head_weight = state_dict.get("head.linear.weight")
    if head_weight is None:
        raise KeyError("Checkpoint state dict missing 'head.linear.weight'")
    head_size = int(head_weight.shape[0])

    model = build_model(num_classes=head_size, variant=variant)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(target_device)

    logger.info(
        "Successfully loaded model from %s (head_size=%d, device=%s)",
        ckpt_path.name,
        head_size,
        target_device,
    )
    return model, head_size


def get_or_load_model(
    display_name: str,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[CustomCNN, List[str], str]:
    """Retrieve model from in-memory cache or load from disk.

    Args:
        display_name: Model name as displayed in UI dropdown.
        device: Target inference device.

    Returns:
        Tuple of (model, known_classes, info_line).
    """
    if display_name in _LOADED_CACHE:
        return _LOADED_CACHE[display_name]

    meta = get_model_metadata(display_name)
    if meta is None:
        raise ValueError(f"Unknown model name: {display_name}")

    ckpt_path = meta.resolve_checkpoint_path()
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint for '{display_name}' not found matching '{meta.pattern}'")

    model, head_size = load_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        device=device,
        variant=meta.variant,
    )

    if head_size != len(meta.known_classes):
        logger.warning(
            "Head size %d in checkpoint %s differs from expected known classes count %d",
            head_size,
            ckpt_path.name,
            len(meta.known_classes),
        )

    result = (model, meta.known_classes, meta.info_line)
    _LOADED_CACHE[display_name] = result
    return result

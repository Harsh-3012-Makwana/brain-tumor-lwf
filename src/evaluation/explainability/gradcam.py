"""Grad-CAM (Gradient-weighted Class Activation Mapping) for Custom CNN.

Computes visual explanations for predictions using the feature activations and
gradients from the final convolutional stage (model.backbone.blocks[3]).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


class GradCAM:
    """Grad-CAM generator for CustomCNN.

    Hooks onto target_layer (model.backbone.blocks[3]) to extract activation maps
    and backward gradients.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to block 3 (the 4th and last conv block) of CustomCNN
            if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
                self.target_layer = model.backbone.blocks[3]
            else:
                raise ValueError("Target layer could not be determined automatically.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self.activations = out
            out.register_hook(self._save_gradient)

        handle = self.target_layer.register_forward_hook(_forward_hook)
        self._hook_handles.append(handle)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate normalized 2D Grad-CAM heatmap for a single image tensor (1, 3, H, W).

        Returns:
            heatmap: (H, W) float array in [0.0, 1.0]
            pred_class: int predicted class index
            pred_prob: float predicted class probability
        """
        self.model.zero_grad()
        logits = self.model(x)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)

        pred_class = int(torch.argmax(logits, dim=1).item())
        if target_class is None:
            target_class = pred_class

        # Ensure target_class is within head bounds
        if target_class >= logits.shape[1]:
            target_class = pred_class

        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")

        # Spatial average of gradients: alpha_k = (1/Z) * sum(grad)
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination: sum(alpha_k * A_k)
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to image size (150, 150)
        h, w = x.shape[2], x.shape[3]
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_upsampled = cam_upsampled.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam_upsampled.min(), cam_upsampled.max()
        if c_max - c_min > 1e-8:
            heatmap = (cam_upsampled - c_min) / (c_max - c_min)
        else:
            heatmap = np.zeros_like(cam_upsampled)

        pred_prob = float(probs[0, pred_class].item())
        return heatmap, pred_class, pred_prob


def overlay_cam_on_image(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Blend a [0, 1] float heatmap with an RGB image [0, 1] using matplotlib colormap."""
    cmap = plt.get_cmap(colormap_name)
    colored_cam = cmap(heatmap)[:, :, :3]  # drop alpha channel -> (H, W, 3)

    # Ensure img_rgb is (H, W, 3) and float in [0, 1]
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0

    overlay = alpha * colored_cam + (1.0 - alpha) * img_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def select_gradcam_test_samples(
    test_csv_path: Path,
    num_per_class: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministically select num_per_class original (non-augmented) images per class."""
    import csv

    rows_by_class: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}
    with open(test_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_str = row["path"]
            # Exclude any augmented images
            if "aug" in path_str.lower():
                continue
            lbl = int(row["global_label"])
            rows_by_class[lbl].append(row)

    selected: List[Dict[str, Any]] = []
    rng = np.random.RandomState(seed)

    for lbl in sorted(rows_by_class.keys()):
        pool = rows_by_class[lbl]
        # Sort first for deterministic seed behavior
        pool.sort(key=lambda r: r["path"])
        chosen_indices = rng.choice(len(pool), size=min(num_per_class, len(pool)), replace=False)
        for idx in sorted(chosen_indices):
            selected.append(pool[idx])

    return selected

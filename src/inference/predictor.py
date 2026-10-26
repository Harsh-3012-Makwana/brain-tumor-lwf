"""Inference and explainability pipeline execution routines.

Provides:
- Predict: Softmax probability distribution computation over known task classes
- Grad-CAM: Visual attention heatmap generation and RGB alpha-blending
- End-to-end unified inference driver for the Gradio interactive demo
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.evaluation.explainability.gradcam import GradCAM, overlay_cam_on_image
from src.inference.preprocessor import preprocess_image
from src.models.custom_cnn.model import CustomCNN

logger = logging.getLogger(__name__)


def predict_probabilities(
    model: CustomCNN,
    tensor: torch.Tensor,
    known_classes: List[str],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Compute softmax class probabilities for a normalized image tensor.

    Args:
        model: CustomCNN model instance in eval mode.
        tensor: Normalized image tensor of shape (1, 3, 150, 150).
        known_classes: List of class names corresponding to output logits.
        device: Device to execute forward pass on.

    Returns:
        Dictionary mapping class names to float probabilities summing to 1.0.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    x = tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Ensure output dictionary matches known classes
    result: Dict[str, float] = {}
    for i, cls_name in enumerate(known_classes):
        if i < len(probs):
            result[cls_name] = float(probs[i])
        else:
            result[cls_name] = 0.0

    return result


def compute_gradcam_overlay(
    model: CustomCNN,
    tensor: torch.Tensor,
    display_image: np.ndarray,
    target_class: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, int, float]:
    """Generate Grad-CAM explanation overlay for the model prediction.

    Args:
        model: CustomCNN model instance.
        tensor: Normalized image tensor (1, 3, 150, 150).
        display_image: Cropped RGB uint8 array (150, 150, 3).
        target_class: Optional class index to explain (defaults to top-1 predicted).
        device: Device to execute Grad-CAM backward pass on.
        alpha: Overlay blend weight between heatmap and raw image.

    Returns:
        Tuple of (overlay_rgb_uint8, pred_class_idx, pred_prob).
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    x = tensor.to(device).clone().detach()

    model.eval()
    cam = GradCAM(model)
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            heatmap, pred_idx, pred_prob = cam.generate(x, target_class=target_class)
    finally:
        cam.remove_hooks()

    # Blend heatmap over the preprocessed display image
    blended_float = overlay_cam_on_image(
        img_rgb=display_image,
        heatmap=heatmap,
        alpha=alpha,
        colormap_name="jet",
    )
    overlay_uint8 = (blended_float * 255.0).clip(0, 255).astype(np.uint8)

    return overlay_uint8, pred_idx, pred_prob


def run_inference_pipeline(
    model: CustomCNN,
    image_input: Union[str, Path, np.ndarray, Image.Image],
    known_classes: List[str],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, str]:
    """Execute complete end-to-end inference and explainability pipeline.

    Args:
        model: CustomCNN model instance.
        image_input: Raw input image (filepath, numpy array, or PIL image).
        known_classes: List of class names for model head.
        device: Device to execute computation on.

    Returns:
        display_image: Preprocessed RGB uint8 image (150, 150, 3).
        probs: Softmax class probabilities dictionary.
        overlay: Grad-CAM heatmap blended on display_image (150, 150, 3).
        prediction_text: Human-readable prediction summary with confidence note.
    """
    # 1. Standard Phase 1 contour crop + resize + replication + normalization
    display_image, normalized_tensor = preprocess_image(image_input)

    # 2. Compute class probabilities
    probs = predict_probabilities(
        model=model,
        tensor=normalized_tensor,
        known_classes=known_classes,
        device=device,
    )

    # 3. Compute Grad-CAM overlay for top predicted class
    pred_class_name = max(probs, key=probs.get)
    pred_class_idx = known_classes.index(pred_class_name)

    overlay, _, top_prob = compute_gradcam_overlay(
        model=model,
        tensor=normalized_tensor,
        display_image=display_image,
        target_class=pred_class_idx,
        device=device,
    )

    # 4. Format prediction summary string
    confidence_pct = top_prob * 100.0
    pred_text = f"Top Prediction: {pred_class_name.upper()} ({confidence_pct:.1f}% confidence)"
    if top_prob < 0.60:
        pred_text += "\n⚠️ Low confidence / uncertain prediction (< 60.0% probability)."

    return display_image, probs, overlay, pred_text

"""Image preprocessing pipeline for brain tumor MRI inference.

Reuses the exact Phase 1 standardization pipeline:
- Largest external contour cropping with Gaussian blur (k=5) and Otsu thresholding
- Margin expansion (2%) and safety fallbacks (area < 20% or touches 4 boundaries)
- Resize to 150x150
- Replicate single-channel grayscale to 3 RGB channels
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.preprocessing.crop import preprocess_image_pipeline

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def load_raw_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Load arbitrary image input into an RGB uint8 numpy array.

    Args:
        image_input: File path (str/Path), numpy array, or PIL Image.

    Returns:
        RGB image as uint8 numpy array of shape (H, W, 3).
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode image file: {img_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)

    if isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            if arr.shape[2] == 1:
                return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
            if arr.shape[2] == 3:
                return arr

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def preprocess_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (150, 150),
    blur_ksize: int = 5,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Execute complete preprocessing and normalization pipeline.

    Args:
        image_input: Raw image path, PIL Image, or uint8 numpy array.
        target_size: Output spatial resolution (default 150x150).
        blur_ksize: Gaussian kernel size for Otsu contour detection (default 5).

    Returns:
        display_image: uint8 numpy array of shape (target_size[0], target_size[1], 3)
                       representing the cropped, resized image seen by the model.
        normalized_tensor: Float32 torch.Tensor of shape (1, 3, target_size[0], target_size[1])
                           normalized with ImageNet mean and std.
    """
    raw_rgb = load_raw_image(image_input)

    # 1. Extreme contour cropping, resizing to target_size, and 3-channel replication
    display_image, fallback_used = preprocess_image_pipeline(
        image=raw_rgb,
        target_size=target_size,
        blur_ksize=blur_ksize,
    )
    if fallback_used:
        logger.debug("Contour crop fallback triggered during inference preprocessing.")

    # 2. Normalize with ImageNet mean and std
    tensor = torch.from_numpy(display_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    normalized_tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    return display_image, normalized_tensor

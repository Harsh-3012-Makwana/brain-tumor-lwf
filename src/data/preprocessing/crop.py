"""Extreme contour cropping, brain isolation, and standardization routines."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def crop_brain_contour(
    image: np.ndarray,
    blur_ksize: int = 5,
    margin_ratio: float = 0.02,
    min_area_ratio: float = 0.20,
) -> Tuple[np.ndarray, bool]:
    """Isolate brain tissue via largest external contour detection.

    Args:
        image: Input image (grayscale 2D or color 3D uint8 array).
        blur_ksize: Kernel dimension for Gaussian smoothing.
        margin_ratio: Fractional bounding box expansion margin (default 0.02 = 2%).
        min_area_ratio: Minimum acceptable contour area relative to image size (default 0.20 = 20%).

    Returns:
        Tuple of (cropped_image, fallback_used).
        If valid contour is detected, returns cropped region and False.
        If detection fails, returns safe fallback (center crop) and True.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty input image provided for contour cropping.")

    h, w = image.shape[:2]
    total_area = float(h * w)

    # 1. Convert to grayscale for thresholding
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image.copy()

    # 2. Gaussian smoothing
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        blurred = gray

    # 3. Otsu automatic binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological erosion and dilation to bridge gaps and remove isolated skull markers
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 5. Extract external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _apply_center_crop(image), True

    # Find contour with maximum area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Fallback 1: Area < 20% of image
    if area < (min_area_ratio * total_area):
        return _apply_center_crop(image), True

    # 6. Extract extreme points
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])

    # Bounding coordinates
    x1, x2 = ext_left[0], ext_right[0]
    y1, y2 = ext_top[1], ext_bot[1]

    # Fallback 2: touches all 4 edges (contour encompasses the entire frame/border artifact)
    touches_left = x1 <= 1
    touches_right = x2 >= (w - 2)
    touches_top = y1 <= 1
    touches_bottom = y2 >= (h - 2)
    if touches_left and touches_right and touches_top and touches_bottom:
        return _apply_center_crop(image), True

    # Apply small expansion margin
    margin_x = int((x2 - x1) * margin_ratio)
    margin_y = int((y2 - y1) * margin_ratio)

    x1 = max(0, x1 - margin_x)
    x2 = min(w, x2 + margin_x)
    y1 = max(0, y1 - margin_y)
    y2 = min(h, y2 + margin_y)

    if (x2 - x1) <= 10 or (y2 - y1) <= 10:
        return _apply_center_crop(image), True

    cropped = image[y1:y2, x1:x2]
    return cropped, False


def _apply_center_crop(image: np.ndarray, crop_fraction: float = 0.95) -> np.ndarray:
    """Safe fallback: central region crop preserving majority of brain structure."""
    h, w = image.shape[:2]
    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y : start_y + crop_h, start_x : start_x + crop_w]


def preprocess_image_pipeline(
    image: np.ndarray,
    target_size: Tuple[int, int] = (150, 150),
    blur_ksize: int = 5,
) -> Tuple[np.ndarray, bool]:
    """Execute complete preprocessing pipeline:
    Grayscale -> Gaussian Blur -> Otsu Threshold -> Largest Contour Crop ->
    Resize (150x150) -> Replicate to 3 Channels -> Return uint8 standardized image.

    Returns:
        Tuple of (standardized_image_3ch, fallback_used).
    """
    # 1. Contour crop
    cropped, fallback_used = crop_brain_contour(image, blur_ksize=blur_ksize)

    # 2. Convert to grayscale if not already
    if cropped.ndim == 3:
        if cropped.shape[2] == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        elif cropped.shape[2] == 4:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cropped[:, :, 0]
    else:
        gray = cropped

    # 3. Resize to target dimension (150, 150)
    interp = cv2.INTER_AREA if gray.shape[0] >= target_size[1] else cv2.INTER_CUBIC
    resized = cv2.resize(gray, target_size, interpolation=interp)

    # 4. Replicate to 3 channels (matching RGB model input option)
    replicated_3ch = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return replicated_3ch, fallback_used

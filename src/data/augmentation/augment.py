"""Data augmentation and normalization pipelines using Albumentations."""

from __future__ import annotations

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Standard ImageNet statistics applied to the 3 replicated grayscale channels
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: Tuple[int, int] = (150, 150),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Return mild training augmentation pipeline.

    Brain MRIs feature subtle tumor boundary margins, so aggressive distortions
    (e.g., elastic deformations, extreme crops) are avoided to maintain pathological fidelity.
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            border_mode=0,
            fill=0,
            p=0.4,
        ),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.2),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_eval_transforms(
    image_size: Tuple[int, int] = (150, 150),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Return deterministic validation and evaluation transform pipeline (Resize + Normalize only)."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])

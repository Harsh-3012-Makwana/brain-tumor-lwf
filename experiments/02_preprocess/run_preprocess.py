"""Runner script for Phase 1 Data Preprocessing, Standardization, and Caching."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Insert project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.crop import preprocess_image_pipeline
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("run_preprocess")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def run_preprocessing() -> Tuple[int, int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Preprocess all images from data/raw and cache them in data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    total_images = 0
    fallback_count = 0
    sample_pairs_by_class: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = {
        cls: [] for cls in CLASS_NAMES
    }

    for split in ["Training", "Testing"]:
        split_raw_dir = RAW_DATA_DIR / split
        split_proc_dir = PROCESSED_DATA_DIR / split

        for cls_name in CLASS_NAMES:
            cls_raw_dir = split_raw_dir / cls_name
            cls_proc_dir = split_proc_dir / cls_name
            cls_proc_dir.mkdir(parents=True, exist_ok=True)

            file_list = sorted([
                f for f in cls_raw_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ])

            logger.info("Processing %s/%s (%d images)...", split, cls_name, len(file_list))

            for file_path in tqdm(file_list, desc=f"{split}/{cls_name}"):
                raw_bgr = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if raw_bgr is None:
                    logger.error("Failed to read image: %s", file_path)
                    continue

                processed_rgb, fallback_used = preprocess_image_pipeline(
                    raw_bgr, target_size=(150, 150), blur_ksize=5
                )

                if fallback_used:
                    fallback_count += 1

                # Save processed RGB image as JPEG / PNG
                out_path = cls_proc_dir / file_path.name
                # cv2 expects BGR for imwrite
                processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), processed_bgr)
                total_images += 1

                # Collect visual samples for comparison grid (up to 4 per class)
                if len(sample_pairs_by_class[cls_name]) < 4:
                    raw_rgb = (
                        cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                        if raw_bgr.ndim == 3
                        else cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2RGB)
                    )
                    sample_pairs_by_class[cls_name].append(
                        (raw_rgb, processed_rgb, file_path.name)
                    )

    fallback_rate = (fallback_count / total_images) * 100 if total_images > 0 else 0.0
    logger.info(
        "Preprocessing finished! Total images: %d, Fallbacks: %d (%.2f%%)",
        total_images,
        fallback_count,
        fallback_rate,
    )

    return total_images, fallback_count, sample_pairs_by_class


def generate_visual_comparison_grid(
    sample_pairs_by_class: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]],
    output_path: Path,
) -> None:
    """Produce a 4x4 visual comparison grid of raw vs cropped images.
    4 rows (one per class), each with 2 pairs of (Raw, Processed).
    Total: 8 columns (4 images across: Raw A, Processed A, Raw B, Processed B) x 4 classes.
    """
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))

    for row_idx, cls_name in enumerate(CLASS_NAMES):
        pairs = sample_pairs_by_class[cls_name]
        # Pair 1: columns 0 and 1
        if len(pairs) > 0:
            raw1, proc1, name1 = pairs[0]
            axes[row_idx, 0].imshow(raw1)
            axes[row_idx, 0].set_title(f"{cls_name.capitalize()} (Raw)\n{raw1.shape[1]}x{raw1.shape[0]}", fontsize=9)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(proc1)
            axes[row_idx, 1].set_title(f"{cls_name.capitalize()} (Processed)\n150x150", fontsize=9)
            axes[row_idx, 1].axis("off")

        # Pair 2: columns 2 and 3
        if len(pairs) > 1:
            raw2, proc2, name2 = pairs[1]
            axes[row_idx, 2].imshow(raw2)
            axes[row_idx, 2].set_title(f"{cls_name.capitalize()} (Raw)\n{raw2.shape[1]}x{raw2.shape[0]}", fontsize=9)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(proc2)
            axes[row_idx, 3].set_title(f"{cls_name.capitalize()} (Processed)\n150x150", fontsize=9)
            axes[row_idx, 3].axis("off")

    fig.suptitle("Preprocessing Verification: Raw MRI vs Contour Cropped & Standardized (150x150)", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved visual comparison figure to %s", output_path)


def main() -> None:
    total, fallbacks, samples = run_preprocessing()
    grid_path = FIGURES_DIR / "preprocessing_check.png"
    generate_visual_comparison_grid(samples, grid_path)


if __name__ == "__main__":
    main()

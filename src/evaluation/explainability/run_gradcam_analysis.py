"""Script to execute Grad-CAM interpretability analysis for Order C Seed 42 models.

Evaluates:
  1. Task 1 Baseline (Shared checkpoint after Task 1: classes {glioma, meningioma})
  2. Naive Fine-Tuning (checkpoint_task2.pt)
  3. LwF (All) (checkpoint_task2.pt)
  4. LwF (New) (checkpoint_task2.pt)
  5. Experience Replay (M=120) (checkpoint_task2.pt)
  6. Joint Training Upper Bound (best.pt)

On 16 deterministic test images (4 per class, original unaugmented scans).
Outputs saved to results/figures/gradcam/
"""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.explainability.gradcam import (
    CLASS_NAMES,
    GradCAM,
    overlay_cam_on_image,
    select_gradcam_test_samples,
)
from src.models.custom_cnn.factory import build_model
from src.utils.paths import FIGURES_DIR, SPLITS_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gradcam_analysis")

GRADCAM_DIR = FIGURES_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    num_classes: int,
    variant: str = "medium",
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Instantiate CustomCNN and load checkpoint weights safely."""
    # Build model with target num_classes
    model = build_model(
        variant=variant,
        num_classes=num_classes,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def find_checkpoint(pattern: str) -> Path:
    """Find a checkpoint matching a glob pattern."""
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching pattern: {pattern}")
    return Path(matches[0])


def get_image_tensor_and_rgb(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Load image, resize to (150, 150), normalize for model, and return RGB float array."""
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((150, 150), Image.BILINEAR)
    img_rgb = np.array(img_resized, dtype=np.float32) / 255.0

    # Model preprocessing transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_resized).unsqueeze(0)  # (1, 3, 150, 150)
    return tensor, img_rgb


def run_gradcam_suite() -> None:
    """Run full Grad-CAM analysis across models and test samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device for Grad-CAM: %s", device)

    # 1. Identify model checkpoints for Order C Seed 42
    model_specs = [
        {
            "name": "Task 1 (Old Only)",
            "num_classes": 2,
            "path": Path("results/checkpoints/shared_task1_order_c_seed42.pt"),
        },
        {
            "name": "Naive Finetune",
            "num_classes": 4,
            "path": find_checkpoint("results/runs/finetune_c_42_*/checkpoint_task2.pt"),
        },
        {
            "name": "LwF (All)",
            "num_classes": 4,
            "path": find_checkpoint("results/runs/lwf_all_c_42_*/checkpoint_task2.pt"),
        },
        {
            "name": "LwF (New)",
            "num_classes": 4,
            "path": find_checkpoint("results/runs/lwf_new_c_42_*/checkpoint_task2.pt"),
        },
        {
            "name": "Replay (M=120)",
            "num_classes": 4,
            "path": find_checkpoint("results/runs/replay120_c_42_*/checkpoint_task2.pt"),
        },
        {
            "name": "Joint (Upper Bound)",
            "num_classes": 4,
            "path": find_checkpoint("results/runs/joint_none_42_medium_*/best.pt"),
        },
    ]

    # Load models
    models = {}
    for spec in model_specs:
        logger.info("Loading model: %s from %s", spec["name"], spec["path"])
        m = load_model_from_checkpoint(
            checkpoint_path=spec["path"],
            num_classes=spec["num_classes"],
            variant="medium",
            device=device,
        )
        models[spec["name"]] = (m, spec["num_classes"])

    # 2. Select 16 test images (4 per class)
    test_csv = SPLITS_DIR / "test.csv"
    samples = select_gradcam_test_samples(test_csv, num_per_class=4, seed=42)
    logger.info("Selected %d test samples for Grad-CAM analysis", len(samples))

    # 3. Compute heatmaps and predictions for all samples and models
    results_by_sample = []
    for s_idx, sample in enumerate(samples):
        img_rel_path = sample["path"]
        img_full_path = PROJECT_ROOT / img_rel_path
        true_label = int(sample["global_label"])
        true_name = CLASS_NAMES[true_label]

        tensor, img_rgb = get_image_tensor_and_rgb(img_full_path)
        tensor = tensor.to(device)

        sample_res = {
            "path": img_rel_path,
            "true_label": true_label,
            "true_name": true_name,
            "img_rgb": img_rgb,
            "model_preds": {},
        }

        for spec in model_specs:
            m_name = spec["name"]
            model, num_cls = models[m_name]

            gradcam = GradCAM(model)
            # Explain predicted class
            heatmap, pred_class, pred_prob = gradcam.generate(tensor)
            overlay = overlay_cam_on_image(img_rgb, heatmap, alpha=0.45)
            gradcam.remove_hooks()

            sample_res["model_preds"][m_name] = {
                "pred_class": pred_class,
                "pred_name": CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else f"C{pred_class}",
                "pred_prob": pred_prob,
                "overlay": overlay,
                "is_correct": (pred_class == true_label),
            }

        results_by_sample.append(sample_res)

    # 4. Generate Master Comparison Grid: 16 rows x (1 raw + 6 models = 7 cols)
    logger.info("Rendering Master Grad-CAM comparison grid...")
    num_rows = len(results_by_sample)
    num_cols = 1 + len(model_specs)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.8, num_rows * 2.8), squeeze=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.35)

    col_titles = ["Original MRI (Truth)"] + [spec["name"] for spec in model_specs]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for r, res in enumerate(results_by_sample):
        # Column 0: Raw image
        ax_raw = axes[r, 0]
        ax_raw.imshow(res["img_rgb"])
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        ax_raw.set_ylabel(f"Img #{r+1}\n[{res['true_name'].upper()}]", fontsize=9, fontweight="bold")

        # Model columns
        for c, spec in enumerate(model_specs, start=1):
            m_name = spec["name"]
            pred_info = res["model_preds"][m_name]
            ax = axes[r, c]
            ax.imshow(pred_info["overlay"])
            ax.set_xticks([])
            ax.set_yticks([])

            # Label subtitle with color
            status_color = "#1b5e20" if pred_info["is_correct"] else "#b71c1c"
            mark = "✓" if pred_info["is_correct"] else "✗"
            subtitle = f"{pred_info['pred_name']} ({pred_info['pred_prob']:.2f}) {mark}"
            ax.set_xlabel(subtitle, fontsize=8.5, fontweight="bold", color=status_color)

    grid_path = GRADCAM_DIR / "gradcam_comparison_grid.png"
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Master Grad-CAM grid to %s", grid_path)

    # 5. Generate Per-Class 4-Sample Panels
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_samples = [r for r in results_by_sample if r["true_label"] == cls_id]
        if not cls_samples:
            continue

        fig, axes = plt.subplots(len(cls_samples), num_cols, figsize=(num_cols * 2.6, len(cls_samples) * 2.6), squeeze=False)
        plt.subplots_adjust(wspace=0.15, hspace=0.35)

        for c, title in enumerate(col_titles):
            axes[0, c].set_title(title, fontsize=10, fontweight="bold", pad=6)

        for r, res in enumerate(cls_samples):
            ax_raw = axes[r, 0]
            ax_raw.imshow(res["img_rgb"])
            ax_raw.set_xticks([])
            ax_raw.set_yticks([])
            ax_raw.set_ylabel(f"Sample {r+1}", fontsize=9, fontweight="bold")

            for c, spec in enumerate(model_specs, start=1):
                m_name = spec["name"]
                pred_info = res["model_preds"][m_name]
                ax = axes[r, c]
                ax.imshow(pred_info["overlay"])
                ax.set_xticks([])
                ax.set_yticks([])

                status_color = "#1b5e20" if pred_info["is_correct"] else "#b71c1c"
                mark = "✓" if pred_info["is_correct"] else "✗"
                subtitle = f"{pred_info['pred_name']} ({pred_info['pred_prob']:.2f}) {mark}"
                ax.set_xlabel(subtitle, fontsize=8, fontweight="bold", color=status_color)

        panel_path = GRADCAM_DIR / f"gradcam_class_{cls_id}_{cls_name}.png"
        plt.savefig(panel_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved class panel to %s", panel_path)

    logger.info("Grad-CAM suite execution finished successfully!")


if __name__ == "__main__":
    run_gradcam_suite()

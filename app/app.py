"""Gradio Interactive Demo for Class-Incremental Brain Tumor MRI Classification.

Allows users to upload or select sample MRI scans, choose from 11 evaluated models
(Joint Upper Bound, Order C Task 1 / Final, Order D Task 1 / Final across Fine-Tuning,
LwF Standard, LwF New-Class CE, and Experience Replay-120), inspect the standardized
preprocessed slice, view class probabilities, and analyze Grad-CAM visual attention overlays.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Disable telemetry and online checks for offline / sandboxed execution
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Localize temporary directories to workspace
_workspace_cache = Path(__file__).resolve().parents[1] / "scratch" / "gradio_temp"
_workspace_cache.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_workspace_cache)
os.environ["GRADIO_EXAMPLES_CACHE"] = str(_workspace_cache / "examples")

import asyncio
import gradio.utils as _gu
import gradio.queueing as _gq
import gradio.routes as _grr

_gu.safe_get_lock = _gq.safe_get_lock = lambda: asyncio.Lock()
_gu.safe_get_stop_event = _grr.safe_get_stop_event = lambda: asyncio.Event()

import gradio as gr
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.explainability.gradcam import select_gradcam_test_samples
from src.inference.predictor import run_inference_pipeline
from src.inference.registry import (
    MODEL_REGISTRY_CONFIGS,
    get_available_model_names,
    get_model_metadata,
    get_or_load_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mri_demo")


def get_default_example_paths() -> List[List[str]]:
    """Retrieve 8 deterministic test examples (2 per class, non-augmented)."""
    test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"
    if test_csv.is_file():
        try:
            samples = select_gradcam_test_samples(test_csv, num_per_class=2, seed=42)
            paths = [[str(PROJECT_ROOT / s["path"])] for s in samples if (PROJECT_ROOT / s["path"]).is_file()]
            if len(paths) >= 8:
                return paths[:8]
        except Exception as e:
            logger.warning("Could not load samples via select_gradcam_test_samples: %s", e)

    # Fallback to direct path search in data/processed/Testing
    fallback_examples: List[List[str]] = []
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    for c in classes:
        folder = PROJECT_ROOT / "data" / "processed" / "Testing" / c
        if folder.is_dir():
            files = sorted([f for f in folder.glob("*.jpg") if "aug" not in f.name.lower()])
            for f in files[:2]:
                fallback_examples.append([str(f)])
    return fallback_examples


def on_model_selection_change(model_name: str) -> str:
    """Return informative metadata line when a model is selected."""
    meta = get_model_metadata(model_name)
    if meta is not None:
        return meta.info_line
    return "Model metadata not available."


def classify_mri(
    image: Optional[np.ndarray],
    model_name: str,
) -> Tuple[Optional[np.ndarray], Dict[str, float], Optional[np.ndarray], str]:
    """Execute classification and Grad-CAM generation for the selected model.

    Args:
        image: Uploaded image as uint8 numpy array (or None).
        model_name: Selected model display name.

    Returns:
        display_img: Standardized preprocessed image seen by the model (150x150).
        probabilities: Class probabilities dictionary.
        cam_overlay: Grad-CAM attention heatmap overlay (150x150).
        prediction_text: Text summarizing prediction, confidence, and uncertainty notes.
    """
    if image is None:
        return (
            None,
            {},
            None,
            "⚠️ Please upload or select an MRI scan image to analyze.",
        )

    if not model_name:
        return (
            None,
            {},
            None,
            "⚠️ Please select a model from the dropdown menu.",
        )

    try:
        model, known_classes, _ = get_or_load_model(model_name)
        display_img, probs, overlay, pred_text = run_inference_pipeline(
            model=model,
            image_input=image,
            known_classes=known_classes,
        )
        return display_img, probs, overlay, pred_text
    except Exception as e:
        logger.exception("Error executing inference on model '%s': %s", model_name, e)
        return (
            None,
            {},
            None,
            f"❌ Inference error: {str(e)}",
        )


def build_app() -> gr.Blocks:
    """Construct and configure the Gradio Blocks application."""
    available_models = get_available_model_names()
    default_model = available_models[0] if available_models else "No models found"
    default_info = on_model_selection_change(default_model) if available_models else ""
    example_images = get_default_example_paths()

    with gr.Blocks(title="MRI Continual Learning Demo", analytics_enabled=False) as demo:
        gr.Markdown(
            """
            # 🧠 Class-Incremental Brain Tumor MRI Classification (Custom CNN)

            > ⚠️ **DISCLAIMER**: **Research prototype for experimentation only. Not a medical device and not for clinical diagnosis or radiological decision-making.**
            
            This application demonstrates how continual learning strategies (Naive Fine-Tuning, Standard LwF, New-Class CE LwF, and Experience Replay) behave on class-incremental tumor identification sequences compared to the Joint Training upper bound.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Model Selection & Configuration")
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="Select Evaluated Checkpoint (Seed 42)",
                    interactive=True,
                )

                model_info = gr.Textbox(
                    value=default_info,
                    label="Model Architecture & Continual Learning Metadata",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("### 2. Input MRI Scan")
                image_input = gr.Image(
                    label="Upload Brain MRI (JPG / PNG)",
                    type="numpy",
                    image_mode="RGB",
                )

                predict_btn = gr.Button("⚡ Analyze & Explain", variant="primary", size="lg")

                if example_images:
                    gr.Markdown("#### Benchmark Example Slices (Non-Augmented Test Set)")
                    gr.Examples(
                        examples=example_images,
                        inputs=[image_input],
                        label="Click any test slice to load:",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 3. Diagnostic & Continual Analysis Outputs")

                prediction_banner = gr.Textbox(
                    label="Prediction & Confidence Assessment",
                    lines=2,
                    interactive=False,
                )

                prob_output = gr.Label(
                    label="Softmax Class Probabilities (Known Classes Only)",
                    num_top_classes=4,
                )

                with gr.Row():
                    preprocessed_output = gr.Image(
                        label="Preprocessed Input (Contour Cropped, 150×150)",
                        type="numpy",
                    )
                    gradcam_output = gr.Image(
                        label="Grad-CAM Attention (Block 4 Overlaid)",
                        type="numpy",
                    )

        # Wire UI event triggers
        model_dropdown.change(
            fn=on_model_selection_change,
            inputs=[model_dropdown],
            outputs=[model_info],
        )

        predict_btn.click(
            fn=classify_mri,
            inputs=[image_input, model_dropdown],
            outputs=[preprocessed_output, prob_output, gradcam_output, prediction_banner],
        )

        # Auto-run when an image is loaded or changed
        image_input.change(
            fn=classify_mri,
            inputs=[image_input, model_dropdown],
            outputs=[preprocessed_output, prob_output, gradcam_output, prediction_banner],
        )

        gr.Markdown(
            """
            ---
            **Continual Learning Insights to Explore:**
            - **Catastrophic Forgetting in Fine-Tuning**: Select *Order C: Fine-Tuning Final* and load a `glioma` or `meningioma` slice (Task 1 old classes). Notice how the model assigns near-zero probability to the true class and collapses predictions into `pituitary` or `notumor` (Task 2 new classes).
            - **Softmax Suppression in Standard LwF**: Select *Order C: LwF (Standard / All CE)*. Observe how all-class cross entropy pushes old logits down, failing to prevent recency bias.
            - **Restricted New-Class CE in LwF**: Select *Order C: LwF (New-Class CE)*. Notice the recovery of old-class probabilities without storing any raw image exemplars.
            - **Task-1 Baseline Head**: Select *Order C: After Task 1* and notice the model output head only knows `glioma` and `meningioma`.
            """
        )

    return demo


def main() -> None:
    """Launch local Gradio web service."""
    demo = build_app()
    logger.info("Starting Gradio demo on http://127.0.0.1:7860 (share=False)")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )


if __name__ == "__main__":
    main()

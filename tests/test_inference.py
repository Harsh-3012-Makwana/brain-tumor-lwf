"""Unit tests for Phase 6 inference pipeline and Gradio application."""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pytest
import torch
from PIL import Image

from app.app import build_app
from src.evaluation.explainability.gradcam import GradCAM, overlay_cam_on_image
from src.inference.predictor import (
    compute_gradcam_overlay,
    predict_probabilities,
    run_inference_pipeline,
)
from src.inference.preprocessor import load_raw_image, preprocess_image
from src.inference.registry import (
    get_available_model_names,
    get_model_metadata,
    get_or_load_model,
    load_model_from_checkpoint,
)
from src.models.custom_cnn.factory import build_model
from src.utils.paths import PROJECT_ROOT


def test_load_raw_image_types():
    """Verify loading from numpy array, PIL Image, and color conversions."""
    # From 2D grayscale numpy
    gray_np = np.zeros((100, 100), dtype=np.uint8)
    rgb = load_raw_image(gray_np)
    assert rgb.shape == (100, 100, 3)
    assert rgb.dtype == np.uint8

    # From 3D RGBA numpy
    rgba_np = np.ones((50, 50, 4), dtype=np.uint8) * 128
    rgb2 = load_raw_image(rgba_np)
    assert rgb2.shape == (50, 50, 3)

    # From PIL Image
    pil_img = Image.fromarray(gray_np)
    rgb3 = load_raw_image(pil_img)
    assert rgb3.shape == (100, 100, 3)


def test_preprocess_image_shapes_and_values():
    """Verify preprocessor returns (150, 150, 3) image and normalized tensor."""
    dummy_input = np.ones((200, 200, 3), dtype=np.uint8) * 100
    display_img, tensor = preprocess_image(dummy_input, target_size=(150, 150))

    assert isinstance(display_img, np.ndarray)
    assert display_img.shape == (150, 150, 3)
    assert display_img.dtype == np.uint8

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 150, 150)
    assert tensor.dtype == torch.float32

    # Normalization should yield non-zero values outside [0, 1] due to ImageNet mean/std
    assert tensor.min() < 0.0 or tensor.max() > 1.0


def test_model_loading_and_head_size():
    """Verify registry loads both 2-class (Task 1) and 4-class (Final) models."""
    available = get_available_model_names()
    assert len(available) >= 11, f"Expected at least 11 available models, found {len(available)}"

    # Load 4-class Joint model
    joint_name = "Joint Training (Upper Bound) - Seed 42"
    model_4cls, classes_4cls, info_4cls = get_or_load_model(joint_name, device="cpu")
    assert model_4cls.head.linear.out_features == 4
    assert len(classes_4cls) == 4
    assert "Joint Upper Bound" in info_4cls

    # Load 2-class Order C Task 1 model
    c_task1_name = "Order C: After Task 1 (glioma, meningioma) - Seed 42"
    model_2cls, classes_2cls, info_2cls = get_or_load_model(c_task1_name, device="cpu")
    assert model_2cls.head.linear.out_features == 2
    assert len(classes_2cls) == 2
    assert classes_2cls == ["glioma", "meningioma"]


def test_predict_probabilities_sum_to_one():
    """Verify predicted probabilities form a valid categorical distribution summing to 1.0."""
    model = build_model(num_classes=4, variant="medium")
    model.eval()

    dummy_tensor = torch.randn(1, 3, 150, 150)
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    probs = predict_probabilities(model, dummy_tensor, classes, device="cpu")

    assert set(probs.keys()) == set(classes)
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-5

    for p in probs.values():
        assert 0.0 <= p <= 1.0


def test_gradcam_overlay_shape_and_bounds():
    """Verify Grad-CAM produces a (150, 150, 3) uint8 image overlay."""
    model = build_model(num_classes=4, variant="medium")
    model.eval()

    dummy_tensor = torch.randn(1, 3, 150, 150)
    dummy_disp = np.ones((150, 150, 3), dtype=np.uint8) * 128

    overlay, pred_idx, pred_prob = compute_gradcam_overlay(
        model=model,
        tensor=dummy_tensor,
        display_image=dummy_disp,
        device="cpu",
    )

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (150, 150, 3)
    assert overlay.dtype == np.uint8
    assert 0 <= pred_idx < 4
    assert 0.0 <= pred_prob <= 1.0


def test_run_inference_pipeline_end_to_end():
    """Verify complete inference pipeline on dummy image array."""
    model = build_model(num_classes=4, variant="medium")
    model.eval()

    dummy_input = np.ones((180, 180, 3), dtype=np.uint8) * 150
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

    disp_img, probs, overlay, pred_text = run_inference_pipeline(
        model=model,
        image_input=dummy_input,
        known_classes=classes,
        device="cpu",
    )

    assert disp_img.shape == (150, 150, 3)
    assert overlay.shape == (150, 150, 3)
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    assert "Top Prediction:" in pred_text


def test_gradio_app_builds_successfully():
    """Verify Gradio application builds without launching."""
    demo = build_app()
    assert demo is not None
    assert hasattr(demo, "blocks")

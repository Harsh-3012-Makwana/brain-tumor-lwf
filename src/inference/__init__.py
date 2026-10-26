"""Inference module for brain tumor MRI classification."""

from src.inference.predictor import (
    compute_gradcam_overlay,
    predict_probabilities,
    run_inference_pipeline,
)
from src.inference.preprocessor import (
    load_raw_image,
    preprocess_image,
)
from src.inference.registry import (
    MODEL_REGISTRY_CONFIGS,
    ModelMetadata,
    get_available_model_names,
    get_model_metadata,
    get_or_load_model,
    load_model_from_checkpoint,
)

__all__ = [
    "compute_gradcam_overlay",
    "get_available_model_names",
    "get_model_metadata",
    "get_or_load_model",
    "load_model_from_checkpoint",
    "load_raw_image",
    "MODEL_REGISTRY_CONFIGS",
    "ModelMetadata",
    "predict_probabilities",
    "preprocess_image",
    "run_inference_pipeline",
]

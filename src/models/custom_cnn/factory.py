"""Model factory and initialization functions for Custom CNN variants."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.models.custom_cnn.model import CustomCNN
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)

# Standard predefined channel configurations for the three width variants
VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small": {
        "stage_channels": [32, 64, 128, 128],
        "convs_per_block": 1,
        "gap_dropout": 0.4,
        "description": "Proposal baseline layout (32, 64, 128, 128), ~0.24M parameters",
    },
    "medium": {
        "stage_channels": [64, 128, 256, 288],
        "convs_per_block": 1,
        "gap_dropout": 0.4,
        "description": "Wider 4-block layout (64, 128, 256, 288), ~1.04M parameters",
    },
    "large": {
        "stage_channels": [64, 160, 384, 544],
        "convs_per_block": 1,
        "gap_dropout": 0.4,
        "description": "High-capacity 4-block layout (64, 160, 384, 544), ~2.53M parameters",
    },
}


def initialize_weights(module: nn.Module) -> None:
    """Apply Kaiming (He) normal initialization to conv layers and unit/zero to BatchNorm."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def build_model(
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    num_classes: int = 4,
    variant: Optional[str] = None,
) -> CustomCNN:
    """Build and initialize a CustomCNN model instance according to config or variant name.

    Args:
        config: Dict of model configurations, or path to YAML config file.
        num_classes: Initial number of classes for the classification head.
        variant: Optional variant name override ('small', 'medium', 'large').

    Returns:
        Fully initialized CustomCNN model instance.
    """
    model_cfg: Dict[str, Any] = {}

    if isinstance(config, (str, Path)):
        loaded = load_yaml(config)
        model_cfg = loaded.get("model", loaded)
    elif isinstance(config, dict):
        model_cfg = config.get("model", config)
    elif config is None and variant is None:
        # Default to loading configs/model/custom_cnn.yaml
        default_yaml = CONFIGS_DIR / "model" / "custom_cnn.yaml"
        if default_yaml.is_file():
            loaded = load_yaml(default_yaml)
            model_cfg = loaded.get("model", loaded)

    # Determine variant
    active_variant = variant or model_cfg.get("active_variant", "small")

    # Check if config has custom variant overrides
    variants_cfg = model_cfg.get("variants", {})
    if active_variant in variants_cfg:
        var_spec = variants_cfg[active_variant]
    elif active_variant in VARIANT_CONFIGS:
        var_spec = VARIANT_CONFIGS[active_variant]
    else:
        raise ValueError(
            f"Unknown variant '{active_variant}'. Supported variants: {list(VARIANT_CONFIGS.keys())}"
        )

    stage_channels = var_spec.get("stage_channels", [32, 64, 128, 128])
    convs_per_block = var_spec.get("convs_per_block", 1)
    dropout_rates = var_spec.get("dropout_rates", [0.0, 0.0, 0.0, 0.0])
    gap_dropout = var_spec.get("gap_dropout", model_cfg.get("head", {}).get("dropout", 0.4))
    in_channels = model_cfg.get("backbone", {}).get("in_channels", 3)

    model = CustomCNN(
        in_channels=in_channels,
        stage_channels=stage_channels,
        convs_per_block=convs_per_block,
        dropout_rates=dropout_rates,
        gap_dropout=gap_dropout,
        num_classes=num_classes,
    )

    # Apply Kaiming initialization to backbone
    initialize_weights(model.backbone)

    logger.debug(
        "Built CustomCNN (variant=%s, stage_channels=%s, num_classes=%d)",
        active_variant,
        stage_channels,
        num_classes,
    )

    return model

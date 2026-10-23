"""Unit tests for YAML configuration loading and merging utilities."""

import pytest
from pathlib import Path
from src.utils.config import load_yaml, merge_configs
from src.utils.paths import CONFIGS_DIR


def test_load_base_config():
    """Verify that the primary base.yaml config loads properly as a non-empty dictionary."""
    base_yaml_path = CONFIGS_DIR / "base.yaml"
    assert base_yaml_path.exists(), f"Expected {base_yaml_path} to exist."

    cfg = load_yaml(base_yaml_path)
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "seeds" in cfg
    assert cfg["seeds"] == [42, 123]
    assert cfg["data"]["image_size"] == [224, 224]


def test_merge_configs():
    """Verify that recursive dictionary merging functions as intended."""
    base = {
        "training": {
            "batch_size": 32,
            "lr": 0.001,
            "optimizer": "adam",
        },
        "model": "custom_cnn",
    }
    override = {
        "training": {
            "lr": 0.0001,
            "weight_decay": 1e-4,
        },
        "notes": "test override",
    }

    merged = merge_configs(base, override)
    assert merged["training"]["batch_size"] == 32
    assert merged["training"]["lr"] == 0.0001
    assert merged["training"]["optimizer"] == "adam"
    assert merged["training"]["weight_decay"] == 1e-4
    assert merged["model"] == "custom_cnn"
    assert merged["notes"] == "test override"


def test_load_nonexistent_file_raises():
    """Verify FileNotFoundError is raised when a non-existent path is requested."""
    with pytest.raises(FileNotFoundError):
        load_yaml(Path("configs/non_existent_config.yaml"))

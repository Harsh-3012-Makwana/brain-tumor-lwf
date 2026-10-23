"""Configuration loading and merging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file safely.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the file is empty or cannot be parsed as a dictionary.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__} in {path}")

    return data


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> Path:
    """Save a dictionary as a YAML configuration file.

    Args:
        data: Dictionary containing configuration parameters.
        file_path: Destination path.

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Args:
        base: Original base configuration dictionary.
        override: New values to override or add to base.

    Returns:
        Deeply merged dictionary.
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = val
    return merged

"""Logging setup and run metadata recording."""

from __future__ import annotations

import datetime
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "brain_tumor_lwf",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure console and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if re-called
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def generate_run_name(method: str, order: str, seed: int) -> str:
    """Generate run directory name according to convention:
    {method}_{order}_{seed}_{timestamp}
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{method}_{order}_{seed}_{timestamp}"


def record_env_info(output_file: Path) -> Dict[str, Any]:
    """Capture environment hardware, OS, Python version, and Git commit hash."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass

    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": git_hash,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = "not_installed"

    with open(output_file, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")

    return info

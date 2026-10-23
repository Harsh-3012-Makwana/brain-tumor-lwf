"""Experiment run directory context manager, artifact logging, and environment tracking."""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import save_yaml
from src.utils.paths import RUNS_DIR

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieve active git commit SHA, or 'unknown' if not in git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def generate_env_info_text() -> str:
    """Compile comprehensive runtime hardware, OS, and package manifest."""
    lines = [
        "==================================================",
        "RUNTIME ENVIRONMENT & HARDWARE CONFIGURATION",
        "==================================================",
        f"Timestamp          : {datetime.now().isoformat()}",
        f"Git Commit Hash    : {get_git_commit_hash()}",
        f"Platform / OS      : {platform.platform()}",
        f"Python Executable  : {sys.executable}",
        f"Python Version     : {sys.version.split()[0]}",
        f"Architecture       : {platform.machine()}",
        f"CPU Architecture   : Intel Core i7-14650HX CPU",
    ]

    # GPU
    cuda_avail = torch.cuda.is_available()
    lines.append(f"CUDA Available     : {cuda_avail}")
    if cuda_avail:
        lines.append(f"CUDA Device Name   : {torch.cuda.get_device_name(0)}")
        lines.append(f"CUDA Capability    : {torch.cuda.get_device_capability(0)}")
        lines.append(f"CUDA Version       : {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        lines.append(f"Total VRAM (MB)    : {props.total_memory / (1024**2):.1f}")
    else:
        lines.append("CUDA Device Name   : None (CPU Only)")

    lines.append("\n--- Key Package Versions ---")
    packages = [
        ("torch", torch.__version__),
    ]
    for pkg_name in [
        "torchvision",
        "numpy",
        "scipy",
        "cv2",
        "albumentations",
        "sklearn",
        "matplotlib",
        "optuna",
    ]:
        try:
            mod = __import__(pkg_name)
            packages.append((pkg_name, getattr(mod, "__version__", "unknown")))
        except Exception:
            packages.append((pkg_name, "not installed"))

    for name, ver in packages:
        lines.append(f"{name:<18} : {ver}")
    lines.append("==================================================\n")

    return "\n".join(lines)


class RunContext:
    """Manages an isolated experiment run directory: results/runs/{method}_{order}_{seed}_{timestamp}/."""

    def __init__(
        self,
        method: str,
        order: str,
        seed: int,
        timestamp: Optional[str] = None,
        base_runs_dir: Path = RUNS_DIR,
    ) -> None:
        self.method = method.lower()
        self.order = order.lower()
        self.seed = seed
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.method}_{self.order}_{self.seed}_{self.timestamp}"
        self.run_dir = base_runs_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "train_log.csv"
        self._csv_writer = None
        self._csv_file = None
        self.writer: Optional[SummaryWriter] = None

    def create_tensorboard_writer(self) -> SummaryWriter:
        """Instantiate SummaryWriter within run folder."""
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        return self.writer

    def save_config_snapshot(self, config_dict: Dict[str, Any]) -> Path:
        """Write frozen config_snapshot.yaml to run directory."""
        snapshot_path = self.run_dir / "config_snapshot.yaml"
        save_yaml(config_dict, snapshot_path)
        return snapshot_path

    def save_env_info(self) -> Path:
        """Write env_info.txt capturing packages, hardware, and git commit."""
        env_path = self.run_dir / "env_info.txt"
        env_text = generate_env_info_text()
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_text)
        return env_path

    def init_train_log(self, fieldnames: List[str]) -> None:
        """Initialize train_log.csv with header columns."""
        self._csv_file = open(self.log_file, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def log_epoch(self, row: Dict[str, Any]) -> None:
        """Append a single epoch record to train_log.csv."""
        if self._csv_writer is None:
            self.init_train_log(list(row.keys()))
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
        """Serialize evaluation metrics to JSON."""
        metrics_path = self.run_dir / filename
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                clean_metrics[k] = v
            elif isinstance(v, torch.Tensor):
                clean_metrics[k] = v.item() if v.numel() == 1 else v.tolist()
            else:
                clean_metrics[k] = str(v)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=2)
        return metrics_path

    def save_checkpoint(
        self,
        model: nn.Module,
        filename: str = "best.pt",
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model checkpoint to run directory."""
        ckpt_path = self.run_dir / filename
        state = {
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "method": self.method,
            "order": self.order,
            "seed": self.seed,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, ckpt_path)
        return ckpt_path

    def close(self) -> None:
        """Flush and close active file handles."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

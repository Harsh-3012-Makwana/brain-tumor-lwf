"""Execution script for Stage E Order D incremental runs.

Order D: T1={notumor, pituitary}, T2={glioma, meningioma} ([2, 3] -> [0, 1])
Two tasks, head grows 2 -> 4.
Task 1 trained ONCE per seed (30 epochs, lr 1e-3) and shared by all methods.
Task 2 trained with lr 3e-4, 30 epochs, standard BN, last-epoch weights.

Methods evaluated:
  - finetune
  - lwf_all
  - lwf_new
  - replay60
  - replay120
Seeds: [42, 123] (10 runs total)
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.incremental.runner.incremental_runner import run_single_incremental_experiment
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, RUNS_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("stage_e_order_d")

# Order D manifest: finetune must run first for each seed to create shared Task 1
ORDER_D_MANIFEST = [
    # Seed 42
    ("finetune", "D", 42),
    ("lwf_all", "D", 42),
    ("lwf_new", "D", 42),
    ("replay60", "D", 42),
    ("replay120", "D", 42),
    # Seed 123
    ("finetune", "D", 123),
    ("lwf_all", "D", 123),
    ("lwf_new", "D", 123),
    ("replay60", "D", 123),
    ("replay120", "D", 123),
]


def get_completed_run_metrics(method: str, order: str, seed: int) -> Optional[Dict[str, Any]]:
    """Check if an incremental run has already completed in results/runs/ and return its metrics."""
    pattern = f"{method.lower()}_{order.lower()}_{seed}_"
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir() and p.name.startswith(pattern):
            metrics_file = p / "metrics.json"
            if metrics_file.is_file():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "average_incremental_accuracy" in data and "final_macro_f1" in data:
                        return data
                except Exception:
                    pass
    return None


def run_order_d_experiments(epochs_per_task: int = 30) -> None:
    """Execute all missing Order D runs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device for Order D: %s", device)

    custom_cnn_yaml = CONFIGS_DIR / "model" / "custom_cnn.yaml"
    model_cfg = load_yaml(custom_cnn_yaml)
    variant = model_cfg.get("active_variant", "medium")
    logger.info("Selected Custom CNN architecture variant: %s", variant.upper())

    for method, order, seed in ORDER_D_MANIFEST:
        existing = get_completed_run_metrics(method=method, order=order, seed=seed)
        if existing is not None:
            logger.info("Resuming: Skipping completed run %s_%s_%d", method, order, seed)
            continue

        logger.info("Starting Order D run: method=%s, seed=%d", method, seed)
        res = run_single_incremental_experiment(
            method=method,
            order=order,
            seed=seed,
            variant=variant,
            epochs_per_task=epochs_per_task,
            device=device,
        )
        del res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All Order D runs completed successfully!")


if __name__ == "__main__":
    run_order_d_experiments(epochs_per_task=30)

"""Standardized directory path resolution for the project."""

from __future__ import annotations

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Top-level paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DATA_DIR = DATA_DIR / "splits"

# Results subdirectories
RUNS_DIR = RESULTS_DIR / "runs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ABLATIONS_DIR = RESULTS_DIR / "ablations"
EXPORTS_DIR = RESULTS_DIR / "exports"


def get_run_dir(method: str, order: str, seed: int, timestamp: str) -> Path:
    """Generate standardized run directory path according to the project convention:
    results/runs/{method}_{order}_{seed}_{timestamp}
    """
    run_name = f"{method}_{order}_{seed}_{timestamp}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    return run_dir

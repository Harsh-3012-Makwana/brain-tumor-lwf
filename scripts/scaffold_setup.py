import os
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

# Directories requiring .gitkeep
DIRECTORIES_TO_KEEP = [
    # configs
    "configs/data",
    "configs/model",
    "configs/training",
    "configs/incremental",
    "configs/tuning",
    # data
    "data/raw",
    "data/interim",
    "data/processed",
    "data/splits",
    # src
    "src/data/audit",
    "src/data/preprocessing",
    "src/data/augmentation",
    "src/data/datasets",
    "src/data/splitting",
    "src/data/tasks",
    "src/models/custom_cnn",
    "src/models/baselines",
    "src/models/heads",
    "src/models/utils",
    "src/training/trainers",
    "src/training/losses",
    "src/training/optimizers",
    "src/training/callbacks",
    "src/training/tuning",
    "src/incremental/strategies",
    "src/incremental/buffers",
    "src/incremental/teachers",
    "src/incremental/runner",
    "src/evaluation/metrics",
    "src/evaluation/incremental_metrics",
    "src/evaluation/efficiency",
    "src/evaluation/explainability",
    "src/evaluation/plots",
    "src/evaluation/reports",
    "src/inference",
    "src/export",
    "src/app",
    "src/utils",
    # experiments
    "experiments/01_data_audit",
    "experiments/02_preprocess",
    "experiments/03_standard_training",
    "experiments/04_hyperparameter_tuning",
    "experiments/05_incremental_learning",
    "experiments/06_ablations",
    "experiments/07_evaluation",
    "experiments/08_export_and_demo",
    # results
    "results/runs",
    "results/checkpoints",
    "results/logs",
    "results/metrics",
    "results/tables",
    "results/figures",
    "results/ablations",
    "results/exports",
    # notebooks, tests, docs, scripts, assets
    "notebooks",
    "tests",
    "docs/architecture",
    "docs/report",
    "docs/references",
    "scripts",
    "assets",
]

# Package directories under src/ and tests/ that need __init__.py
INIT_DIRECTORIES = [
    "src",
    "src/data",
    "src/data/audit",
    "src/data/preprocessing",
    "src/data/augmentation",
    "src/data/datasets",
    "src/data/splitting",
    "src/data/tasks",
    "src/models",
    "src/models/custom_cnn",
    "src/models/baselines",
    "src/models/heads",
    "src/models/utils",
    "src/training",
    "src/training/trainers",
    "src/training/losses",
    "src/training/optimizers",
    "src/training/callbacks",
    "src/training/tuning",
    "src/incremental",
    "src/incremental/strategies",
    "src/incremental/buffers",
    "src/incremental/teachers",
    "src/incremental/runner",
    "src/evaluation",
    "src/evaluation/metrics",
    "src/evaluation/incremental_metrics",
    "src/evaluation/efficiency",
    "src/evaluation/explainability",
    "src/evaluation/plots",
    "src/evaluation/reports",
    "src/inference",
    "src/export",
    "src/app",
    "src/utils",
    "tests",
]

# Notebook files
NOTEBOOK_FILES = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocessing_check.ipynb",
    "notebooks/03_results_analysis.ipynb",
]

def make_empty_notebook():
    return {
        "cells": [],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def main():
    print(f"Scaffolding directories in {ROOT}...")
    for rel_path in DIRECTORIES_TO_KEEP:
        d = ROOT / rel_path
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    for rel_path in INIT_DIRECTORIES:
        d = ROOT / rel_path
        d.mkdir(parents=True, exist_ok=True)
        init_py = d / "__init__.py"
        if not init_py.exists():
            init_py.write_text(f'"""Package module for {rel_path}."""\n', encoding="utf-8")

    for rel_path in NOTEBOOK_FILES:
        nb_path = ROOT / rel_path
        if not nb_path.exists():
            nb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(make_empty_notebook(), f, indent=2)
            print(f"Created notebook placeholder: {rel_path}")

    print("Directory scaffold, .gitkeep files, and __init__.py files created successfully.")

if __name__ == "__main__":
    main()

# Helper Scripts (`scripts/`)

This directory contains command-line automation scripts, setup helpers, environment verification utilities, and execution shortcuts.

## Contents
- `scaffold_setup.py`: Generates the standardized directory hierarchy, `.gitkeep` markers, and package `__init__.py` files.
- `setup_env.ps1`: Automated environment setup script for Windows PowerShell (creates virtual environment, installs dependencies, verifies PyTorch CUDA).
- `run_all.sh`: Automation script orchestrating sequential pipeline phases from data preparation to evaluation and export.
- `clean.py`: Utility to purge temporary cache files and intermediate checkpoints while preserving critical logs and final results.

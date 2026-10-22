#!/usr/bin/env bash
# Automation runner script for brain tumor LwF pipeline
set -e

echo "=== Brain Tumor LwF Execution Runner ==="

VENV_PYTHON=".venv/Scripts/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON=".venv/bin/python"
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not detected. Initializing setup..."
    python -m venv .venv
    $VENV_PYTHON -m pip install --upgrade pip setuptools wheel
    $VENV_PYTHON -m pip install -r requirements.txt
fi

echo "Running environment and unit test checks..."
$VENV_PYTHON -m pytest tests/ -v

echo "Ready for Phase 1 data audit."

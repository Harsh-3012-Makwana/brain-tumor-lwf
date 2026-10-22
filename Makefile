.PHONY: help setup test lint clean audit preprocess train evaluate demo

PYTHON = .venv/Scripts/python
PIP = .venv/Scripts/pip
PYTEST = .venv/Scripts/pytest

help:
	@echo "Multi-Class Brain Tumor MRI Classification with LwF"
	@echo "Available commands:"
	@echo "  make setup       : Create virtual environment and install dependencies"
	@echo "  make test        : Run pytest suite"
	@echo "  make audit       : Run Phase 1 data audit"
	@echo "  make preprocess  : Run Phase 1 contour cropping and preprocessing"
	@echo "  make train       : Run Phase 5 incremental training runs"
	@echo "  make evaluate    : Run Phase 6 evaluation and report generation"
	@echo "  make demo        : Launch Phase 7 interactive Gradio demo"
	@echo "  make clean       : Purge temporary caches and pycache"

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

audit:
	$(PYTHON) experiments/01_data_audit/run_audit.py

preprocess:
	$(PYTHON) experiments/02_preprocess/run_preprocess.py

train:
	$(PYTHON) experiments/05_incremental_learning/run_incremental.py

evaluate:
	$(PYTHON) experiments/07_evaluation/run_evaluation.py

demo:
	$(PYTHON) src/app/app.py

clean:
	powershell -Command "Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force; Get-ChildItem -Recurse -Filter '*.pyc' | Remove-Item -Force"

# Source Codebase (`src/`)

This directory contains the production-grade modular Python package for all data handling, model definitions, training routines, incremental learning strategies, evaluation metrics, and deployment pipelines.

## Module Breakdown

- `data/`:
  - `audit/`: Dataset verification, class distribution analysis, hash-based near-duplicate detection, and train/test leakage scans.
  - `preprocessing/`: Extreme contour finding, skull-stripping/cropping, Otsu thresholding, bilateral filtering, resizing, and normalization.
  - `augmentation/`: Albumentations pipelines tailored for brain MRI (rotations, flips, affine transforms, Gaussian blur, contrast adjustments).
  - `datasets/`: PyTorch `Dataset` implementations for single tasks, replay buffers, and multi-class evaluations.
  - `splitting/`: Stratified split generation preserving class distributions across seeds.
  - `tasks/`: Class-incremental task builder assigning classes to sequential steps ($T_1, T_2, T_3$).

- `models/`:
  - `custom_cnn/`: 4-block custom lightweight CNN (<5M parameters, target roughly 1M to 3M) with Conv2D, BatchNorm, ReLU, MaxPool, and Global Average Pooling.
  - `heads/`: Expandable classifier heads supporting dynamic output expansion across incremental tasks.
  - `utils/`: Parameter counting, FLOPs calculation, and architectural profiling utilities.

- `training/`:
  - `trainers/`: Base and task-specific training loops.
  - `losses/`: Standard cross-entropy and modified distillation loss for LwF.
  - `optimizers/`: Factories for AdamW, SGD, and learning rate schedulers.
  - `callbacks/`: Checkpointing, early stopping, and metric logging hooks.
  - `tuning/`: Optional Optuna objective runners for hyperparameter exploration.

- `incremental/`:
  - `strategies/`: Continual learning methods evaluated on the custom CNN: Naive Fine-Tuning (lower bound), Learning without Forgetting (LwF; main method), Experience Replay (fixed buffers of 60 and 120 images with class-balanced random selection), and Joint Training (upper bound oracle).
  - `buffers/`: Exemplar replay memory managers (total size 60 or 120 images, class-balanced random selection filled from each finished task).
  - `teachers/`: Snapshotting and inference through frozen previous-task teacher models.
  - `runner/`: Orchestrator executing the complete sequence of incremental tasks.

- `evaluation/`:
  - `metrics/`: Standard classification metrics: Accuracy, Precision, Recall, Macro F1, ROC-AUC.
  - `incremental_metrics/`: Continual learning metrics: Average Incremental Accuracy ($A_{\text{avg}}$), Forgetting ($F$), and Backward Transfer ($\text{BWT}$).
  - `efficiency/`: Model profiling (parameter count, FLOPs, model size in MB, CPU/GPU latency, and replay buffer memory cost in stored images).
  - `explainability/`: Grad-CAM heatmaps comparing attention drift across incremental learning steps.
  - `plots/`: Automated plotting for confusion matrices, task accuracy curves, and ablation charts.
  - `reports/`: Aggregation and statistical reporting (mean $\pm$ std across seeds).

- `inference/`: End-to-end inference pipeline for single and batched MRI slice classification.
- `export/`: ONNX export, shape validation, and runtime optimization.
- `app/`: Interactive Gradio web application for model demonstration and Grad-CAM visualization.
- `utils/`: Core utilities for reproducibility (seed setting), YAML configuration loading, path resolution, and logging.

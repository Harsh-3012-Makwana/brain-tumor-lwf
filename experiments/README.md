# Experiments Runner Directory (`experiments/`)

This directory contains executable entry-point scripts organized systematically by experimental phase. Each script parses configuration files, executes a discrete stage of the pipeline, logs artifacts, and records run snapshots.

## Stages

- `01_data_audit/`: Run dataset inspection, verify image integrity, check class distributions, and run perceptual hash checks for train/test data leakage.
- `02_preprocess/`: Execute extreme contour cropping, apply standardized resizing and normalization, and generate cached processed datasets.
- `03_standard_training/`: Train non-incremental Joint Training upper bounds on all 4 classes for both Custom CNN and ResNet18.
- `04_hyperparameter_tuning/`: Optional Optuna hyperparameter optimization studies for learning rate, weight decay, distillation temperature ($T$), and loss weight ($\lambda$).
- `05_incremental_learning/`: Run sequential multi-task class-incremental experiments across Task Orders A and B and seeds (`42`, `123`) for both Custom CNN and ResNet18:
  - Naive Fine-Tuning (lower bound)
  - Learning without Forgetting (LwF; main method)
  - Experience Replay (fixed buffers of 60 and 120 images)
  - Joint Training (upper bound oracle)
- `06_ablations/`: Execute ablation studies assessing the impact of distillation temperature $T \in \{1, 2, 4, 8\}$, loss weight $\lambda \in \{0.5, 1.0, 2.0, 5.0\}$, replay buffer size (60 vs 120 images), and task presentation orders (Order A vs Order B).
- `07_evaluation/`: Compute continual learning metrics (Average Incremental Accuracy, Forgetting, BWT), generate comparative figures, compute FLOPs/latency, and render Grad-CAM visualizations.
- `08_export_and_demo/`: Export top-performing checkpoints to ONNX format, validate output parity, and launch the interactive Gradio demo application.

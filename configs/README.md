# Configuration System (`configs/`)

This directory houses all experiment, data, model, training, incremental learning, and hyperparameter tuning configurations. 
**No experiment settings or hyperparameters are hardcoded in application code.**

## Directory Structure

- `base.yaml`: Global experiment defaults (seed lists, image resolution, project paths, logging rates).
- `data/`: Dataset definitions, preprocessing parameters (contour crop, thresholds), augmentation pipelines, and train/val/test split specifications.
- `model/`: Architectural definitions for the Custom CNN (<5M parameters, target 1M-3M, main model) and ResNet18 (ImageNet pretrained, reference model), with expandable classification heads.
- `training/`: Optimization hyperparameters (AdamW/SGD, learning rates, weight decay), LR schedulers (CosineAnnealingLR), epochs, and early stopping patience.
- `incremental/`: Configurations for incremental learning strategies:
  - Naive Fine-Tuning (`finetune.yaml`)
  - Learning without Forgetting (`lwf.yaml`) — distillation temperature ($T$) and loss weight ($\lambda$)
  - Experience Replay (`replay.yaml`) — fixed total buffer of 60 and 120 images with class-balanced random selection
  - Joint Training upper bound (`joint.yaml`)
  - Class-incremental task order definitions (`orders.yaml`) for Order A and Order B
- `tuning/`: Optuna search space definitions and optimization study settings.

# Configuration System (`configs/`)

This directory houses all experiment, data, model, training, incremental learning, and hyperparameter tuning configurations. 
**No experiment settings or hyperparameters are hardcoded in application code.**

## Directory Structure

- `base.yaml`: Global experiment defaults (seed lists, image resolution, project paths, logging rates).
- `data/`: Dataset definitions, preprocessing parameters (contour crop, thresholds), augmentation pipelines, and train/val/test split specifications.
- `model/`: Architectural definitions for the Custom CNN (<5M parameters), ResNet18, and VGG16 baselines, and classification head dimensions.
- `training/`: Optimization hyperparameters (Adam/SGD, learning rates, weight decay), LR schedulers (CosineAnnealingLR, ReduceLROnPlateau), epochs, and early stopping patience.
- `incremental/`: Configurations for incremental learning strategies:
  - Naive Fine-Tuning (`finetune.yaml`)
  - Learning without Forgetting (`lwf.yaml`) — distillation temperature ($T$) and loss weight ($\lambda$)
  - Experience Replay (`replay.yaml`) — memory buffer size and sampling strategy
  - Joint Training upper bound (`joint.yaml`)
  - Class-incremental task order definitions (`orders.yaml`) for Order A and Order B
- `tuning/`: Optuna search space definitions and optimization study settings.

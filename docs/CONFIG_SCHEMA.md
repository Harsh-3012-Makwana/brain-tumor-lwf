# Configuration Schema Documentation (`docs/CONFIG_SCHEMA.md`)

This document defines the schema, field types, and default values for all experiment configurations located in `configs/`.

---

## 1. Root Configuration Schema (`configs/base.yaml`)

```yaml
project:
  name: str               # Unique project identifier ("brain-tumor-lwf")
  description: str        # Brief project description
  version: str            # Semantic version string
  device: str             # Compute device: "cuda" or "cpu"
  num_workers: int        # DataLoader multiprocessing workers (default: 4)
  pin_memory: bool        # Fast GPU memory pinning (default: true)

seeds: list[int]          # Active seeds for statistical replication: [42, 123] (seed 2024 is marked unused if split files exist)

paths:
  raw_data_dir: str       # "data/raw"
  interim_data_dir: str   # "data/interim"
  processed_data_dir: str # "data/processed"
  splits_dir: str         # "data/splits"
  results_dir: str        # "results"
  runs_dir: str           # "results/runs"
  checkpoints_dir: str    # "results/checkpoints"
  logs_dir: str           # "results/logs"
  tables_dir: str         # "results/tables"
  figures_dir: str        # "results/figures"
  ablations_dir: str      # "results/ablations"
  exports_dir: str        # "results/exports"

data:
  dataset_name: str       # Benchmark identifier
  num_total_classes: int  # Total diagnostic categories (4)
  class_names: list[str]  # ["glioma", "meningioma", "notumor", "pituitary"]
  global_label_map: dict  # {glioma: 0, meningioma: 1, notumor: 2, pituitary: 3}
  image_size: list[int]   # Input spatial resolution: [height, width] = [150, 150]
  channels: int           # Channel count: 3 (RGB)
  split_ratios:
    val_from_train_ratio: float  # Stratified split fraction from train set (0.15)
  preprocessing:
    apply_contour_crop: bool     # Extreme contour detection cropping (true)
    blur_kernel_size: int        # Gaussian blur kernel size (5)
    threshold_otsu: bool         # Automatic Otsu binarization (true)
    normalize:
      mean: list[float]          # Channel normalization means [0.485, 0.456, 0.406]
      std: list[float]           # Channel normalization stds [0.229, 0.224, 0.225]

training:
  batch_size: int         # Mini-batch size (64)
  num_epochs: int         # Maximum training epochs per task (40)
  gradient_clip_val: float# Max L2 gradient norm clipping (1.0)
  num_workers: int        # DataLoader worker processes (4; fallback 0 on Windows)
  use_amp: bool           # Mixed precision training with bfloat16/fp16 (true)
  bn_policy: str          # BatchNorm continual policy: "standard" | "frozen_after_task1"
  optimizer:
    name: str             # "adamw"
    lr: float             # Base learning rate (0.001 / 1e-3)
    weight_decay: float   # L2 weight regularization (0.0001 / 1e-4)
    betas: list[float]    # AdamW momentum coefficients: [0.9, 0.999]
  scheduler:
    name: str             # "cosine" (Linear Warmup + Cosine Annealing)
    warmup_epochs: int    # Initial warmup epochs (2)
    min_lr: float         # Minimum decayed learning rate floor (1e-6)
  early_stopping:
    patience: int         # Non-improving validation epochs before stopping (8)
    min_delta: float      # Minimum metric increase to register improvement (0.0005)
    monitor: str          # Monitored validation metric ("val_macro_f1")
    mode: str             # Optimization direction ("max")

incremental:
  strategy: str           # "finetune", "lwf", "replay", or "joint"
  active_order: str       # "order_a" or "order_b"
  task_orders: dict       # Mapping of sequential tasks to class subsets
  lwf:
    temperature: float    # Softmax temperature scaling T (2.0)
    lambda_distill: float # Distillation loss scaling factor lambda (1.0)
  replay:
    buffer_sizes: list[int]# Buffer variants: [60, 120] total images
    total_capacity: int   # Active buffer size: 60 or 120
    # Buffer Rule: Total capacity across all old classes, class-balanced random selection,
    # filled incrementally from the data of each finished task.
    # Memory Cost Metric: Reported as number of stored images (60 or 120) next to accuracy & forgetting.
    sampling_strategy: str# "class_balanced_random"

model:
  name: str               # "custom_cnn" (sole model architecture in project)
  hard_param_limit: int   # Strict upper bound (5000000; target roughly 1M to 3M)
  active_variant: str     # Active width variant: "small" (0.24M), "medium" (1.04M), or "large" (2.53M)
  variants:
    small:
      stage_channels: list[int]  # [32, 64, 128, 128] (~0.24M params, proposal baseline)
      convs_per_block: int       # 1
      gap_dropout: float         # 0.4
    medium:
      stage_channels: list[int]  # [64, 128, 256, 288] (~1.04M params)
      convs_per_block: int       # 1
      gap_dropout: float         # 0.4
    large:
      stage_channels: list[int]  # [64, 160, 384, 544] (~2.53M params)
      convs_per_block: int       # 1
      gap_dropout: float         # 0.4
  head:
    type: str             # "expandable_linear"
    initial_num_classes: int # Initial output size (e.g., 2 for Task 1)
    total_classes: int    # Cumulative output size (4)
    dropout: float        # Post-GAP dropout rate (0.4)
```

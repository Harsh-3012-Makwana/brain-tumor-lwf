# Results Directory (`results/`)

This directory collects all experimental outputs, run logs, serialized checkpoints, evaluation summaries, plots, and deployment assets.

## Directory Structure

- `runs/`: Contains isolated run directories following the strict naming schema:
  `{method}_{order}_{seed}_{timestamp}/`
  Every run folder captures:
  - `config_snapshot.yaml`: Complete, frozen YAML configuration used for the run.
  - `train_log.csv`: Epoch-by-epoch loss and metric progression.
  - `events.out.tfevents.*`: TensorBoard event log.
  - `metrics.json`: Per-task and final evaluation metrics.
  - `checkpoints/`: Model weights (`best.pt`, `last.pt`, and per-task checkpoints `task_1.pt`, `task_2.pt`, `task_3.pt`).
  - `predictions/`: Raw predictions (logits and ground truth labels) for confusion matrix compilation.
  - `env_info.txt`: Hardware specifications, environment packages, and Git commit hash.
- `checkpoints/`: Curated best final checkpoints for each experimental strategy across seeds.
- `logs/`: Centralized or aggregated TensorBoard log directory.
- `metrics/`: Standalone JSON and CSV evaluation summaries across all experimental runs.
- `tables/`: Publication-ready comparison tables (Markdown and CSV) aggregating performance across seeds (mean $\pm$ std).
- `figures/`: High-resolution figures (confusion matrices, task accuracy trajectories, forgetting bar charts, Grad-CAM overlays).
- `ablations/`: Output figures, metrics, and logs specific to hyperparameter and sensitivity ablations.
- `exports/`: Optimized ONNX model weights and latency profiling benchmarks.

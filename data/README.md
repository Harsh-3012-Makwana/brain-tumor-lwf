# Data Management (`data/`)

This directory manages all raw, intermediate, processed, and partitioned split data for the brain tumor MRI experiments.

## Directory Structure

- `raw/`: Immutable raw dataset storage. The original Masoud Nickparvar Brain Tumor MRI Dataset (7,023 images across 4 classes: `glioma`, `meningioma`, `pituitary`, `notumor`) must be placed here. **Never modify files in `raw/`**.
- `interim/`: Intermediate artifacts generated during data auditing and exploratory cleaning (e.g., duplicate detection reports, image corruption logs, intensity histogram dumps).
- `processed/`: Standardized, cropped (via extreme contour detection), resized, and normalized images cached for rapid dataloader throughput.
- `splits/`: Deterministic, stratified split indices generated per active seed (`42`, `123`; seed `2024` splits are marked unused):
  - Train (stratified 85% of official train) / Val (stratified 15% of official train) / Test (official 1,311 test set).
  - Class-incremental task partitions for Task Order A and Task Order B.

## Storage and Versioning Policy
Raw image binaries and processed caches are excluded from version control via `.gitignore`. Split manifest files (CSV/JSON/YAML) in `splits/` are version-controlled to ensure 100% reproducible experiments across environments.

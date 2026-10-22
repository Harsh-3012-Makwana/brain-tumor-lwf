# Exploration Notebooks (`notebooks/`)

This directory is strictly reserved for exploratory data analysis, visual inspection of preprocessing pipelines, and post-hoc evaluation analysis.

## Policy and Best Practices
- **No Production Logic**: Code in this directory is for scratchpad experimentation and inspection only. Production logic must reside in `src/`.
- **No Downstream Imports**: No module under `src/` or `experiments/` may import code from `notebooks/`.
- **Reproducible State**: Notebooks must be cleared of unnecessary scratch state before committing, or kept clean and self-contained.

## Pre-Created Notebooks
- `01_eda.ipynb`: Exploratory data analysis, slice dimensions, class distribution, and intensity profiling of the raw Nickparvar MRI dataset.
- `02_preprocessing_check.ipynb`: Visual validation of extreme contour detection, bounding-box cropping, artifact removal, and Albumentations transforms.
- `03_results_analysis.ipynb`: Interactive visualization of continual learning curves, forgetting analysis, confusion matrices, and Grad-CAM attention shifts.

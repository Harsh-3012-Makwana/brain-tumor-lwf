# Multi-Class Brain Tumor Classification in MRI Using a Lightweight CNN (<5M Parameters) and Class-Incremental Learning with Learning without Forgetting (LwF)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Abstract
*Abstract placeholder — to be updated upon completion of Phase 6 evaluation.*

In clinical neuro-oncology, magnetic resonance imaging (MRI) diagnostic models encounter continuous domain and class shifts as newly characterized tumor sub-types or institutional imaging protocols emerge. Storing historical patient scans to periodically retrain models from scratch is heavily restricted by data privacy regulations (e.g., HIPAA, GDPR) and local storage constraints. 

**Core Research Question**:  
> *Can a lightweight custom CNN (<5M params) retain old tumor classes under class-incremental learning with LwF, and how does it compare with naive fine-tuning, tiny-buffer experience replay (60 and 120 images), and joint training?*

Using the 7,023-image Masoud Nickparvar Brain Tumor MRI dataset across 4 classes (`glioma`, `meningioma`, `pituitary`, `notumor`), this research evaluates continual learning methods exclusively on **one single model architecture**:
- **Custom CNN**: Built from scratch, strictly $<5\text{M}$ parameters (target range roughly 1M to 3M; 4 convolutional blocks with BatchNorm, ReLU, MaxPool; followed by Global Average Pooling and an expandable classification head).

We benchmark 4 incremental learning methods plus Joint Training across two task presentation sequences (Order A and Order B) and two deterministic seeds (`42`, `123`), producing a focused 18-run experimental matrix.

---

## Repository Structure

```
brain-tumor-lwf/
├── configs/                          # Experiment configs (base, data, model, training, incremental, tuning)
├── data/
│   ├── raw/                          # Original Nickparvar dataset (never modified)
│   ├── interim/                      # Temporary outputs (audit reports, duplicate checks)
│   ├── processed/                    # Cropped, resized, cached MRI images
│   └── splits/                       # Stratified train/val/test and incremental task manifests
├── src/
│   ├── data/                         # Audit, contour preprocessing, augmentations, datasets, splitting, tasks
│   ├── models/                       # Custom lightweight CNN (<5M params), expandable heads, profiling utils
│   ├── training/                     # Standard trainers, CE/distillation losses, optimizers, callbacks, tuning
│   ├── incremental/                  # Continual learning strategies (LwF, fine-tuning, replay, joint, teachers)
│   ├── evaluation/                   # Metrics (Accuracy, F1, AIA, Forgetting, BWT), FLOPs/latency, Grad-CAM
│   ├── inference/                    # Single/batch inference pipeline
│   ├── export/                       # ONNX export and runtime validation
│   ├── app/                          # Interactive Gradio demonstration web application
│   └── utils/                        # Seed determinism, YAML config loader, path definitions, logging
├── experiments/                      # Modular phase entry-point scripts (01 to 08)
├── results/
│   ├── runs/                         # Isolated run outputs ({method}_{order}_{seed}_{timestamp})
│   ├── checkpoints/                  # Curated best model weights per strategy
│   ├── logs/                         # TensorBoard event logs
│   ├── metrics/                      # Serialized metric JSON/CSV summaries
│   ├── tables/                       # Publication-ready Markdown/CSV tables (mean ± std)
│   ├── figures/                      # Confusion matrices, curves, Grad-CAM overlays
│   ├── ablations/                    # Ablation logs and sensitivity plots
│   └── exports/                      # Optimized ONNX model graphs
├── notebooks/                        # Exploratory analysis only (EDA, preprocessing checks, results analysis)
├── tests/                            # Unit and integration test suites mirroring src/
├── docs/                             # Plan, decision log, experiment matrix, metrics definitions, references
├── scripts/                          # Automated setup and pipeline shell scripts
├── assets/                           # Graphics, demo clips, sample MRI slices
├── .gitignore
├── requirements.txt
├── README.md
└── Makefile / run.sh
```

---

## Setup & Installation

### 1. Prerequisites
- **Operating System**: Windows 11 / Linux
- **Python**: Version 3.11 (3.11.9 recommended)
- **CUDA**: NVIDIA GPU with CUDA 12.4+ driver support (or CPU fallback)

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd brain-tumor-lwf

# Create virtual environment with Python 3.11
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Verification
Verify that PyTorch detects your GPU and all testing suites pass:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
pytest tests/ -v
```

---

## Dataset Organization

Download the **Masoud Nickparvar Brain Tumor MRI Dataset** and place it directly into `data/raw/` preserving the original directory layout:

```
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

*Note: The official `Testing/` partition (1,311 images) is strictly held out for final evaluation. The active training and validation sets are generated deterministically using stratified 15% partitioning.*

---

## Incremental Task Orders

Models are evaluated across two distinct 3-task class-incremental presentation sequences:
- **Order A (Pathology-First)**:
  - Task 1 ($T_1$): `{glioma, meningioma}`
  - Task 2 ($T_2$): `{pituitary}`
  - Task 3 ($T_3$): `{notumor}`
- **Order B (Control-First)**:
  - Task 1 ($T_1$): `{notumor, pituitary}`
  - Task 2 ($T_2$): `{glioma}`
  - Task 3 ($T_3$): `{meningioma}`

---

## Comparative Results (Mean ± Std over Seeds [42, 123])

Complete experimental evaluations across all four incremental sequences (Orders A, B, C, D) and Joint Training are documented in `results/tables/final_summary.md` and `results/tables/incremental_results.md`.

| Order | Method | Replay Buffer | Teacher Model | $A_{\text{avg}}$ | Final Accuracy | Final Macro-F1 | Forgetting ($F$) ↓ | Old-Class Acc | New-Class Acc |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **All** | **Joint Training (Upper Bound)** | 0 images | — | **—** | **0.8862 ± 0.0056** | **0.8831 ± 0.0057** | **0.0000** | — | — |
| **C** | **FINETUNE** | 0 images | — | 0.6865 ± 0.0016 | 0.4994 ± 0.0006 | 0.3331 ± 0.0006 | 0.9688 ± 0.0006 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| **C** | **LWF (Standard / All CE)** | 0 images | 3.96 MB | 0.6862 ± 0.0003 | 0.4994 ± 0.0006 | 0.3331 ± 0.0006 | 0.9675 ± 0.0006 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| **C** | **LWF_NEW (New-Class CE)** | 0 images | 3.96 MB | 0.7349 ± 0.0125 | 0.6022 ± 0.0141 | 0.5888 ± 0.0140 | **0.4681 ± 0.0094** | 0.3994 ± 0.0094 | 0.8050 ± 0.0187 |
| **C** | **REPLAY60** | 60 images | — | 0.7584 ± 0.0084 | 0.6288 ± 0.0162 | 0.5739 ± 0.0210 | 0.6075 ± 0.0318 | 0.2581 ± 0.0318 | 1.0000 ± 0.0000 |
| **C** | **REPLAY120** | 120 images | — | 0.7834 ± 0.0075 | 0.6993 ± 0.0106 | 0.6720 ± 0.0135 | 0.4688 ± 0.0212 | 0.3987 ± 0.0212 | 1.0000 ± 0.0000 |
| **D** | **FINETUNE** | 0 images | — | 0.6980 ± 0.0028 | 0.4144 ± 0.0025 | 0.2930 ± 0.0026 | 0.9988 ± 0.0013 | 0.0000 ± 0.0000 | 0.8288 ± 0.0050 |
| **D** | **LWF (Standard / All CE)** | 0 images | 3.96 MB | 0.6977 ± 0.0022 | 0.4150 ± 0.0025 | 0.2933 ± 0.0028 | 0.9969 ± 0.0006 | 0.0000 ± 0.0000 | 0.8300 ± 0.0050 |
| **D** | **LWF_NEW (New-Class CE)** | 0 images | 3.96 MB | 0.7601 ± 0.0044 | 0.5912 ± 0.0106 | 0.5639 ± 0.0246 | **0.1431 ± 0.0093** | 0.8569 ± 0.0093 | 0.3251 ± 0.0312 |
| **D** | **REPLAY60** | 60 images | — | 0.7788 ± 0.0000 | 0.6256 ± 0.0069 | 0.6038 ± 0.0100 | 0.5694 ± 0.0206 | 0.4306 ± 0.0206 | 0.8206 ± 0.0069 |
| **D** | **REPLAY120** | 120 images | — | 0.8172 ± 0.0019 | 0.7275 ± 0.0056 | 0.7289 ± 0.0124 | 0.3557 ± 0.0181 | 0.6443 ± 0.0181 | 0.8106 ± 0.0069 |

### Custom CNN Architectural Efficiency Profile (Medium Variant)
- **Parameter Count**: 1,036,548 (<5,000,000 constraint)
- **FLOPs (Input: 3 × 150 × 150)**: 1,077.56 MFLOPs (1.08 GFLOPs)
- **Disk Footprint (Checkpoint)**: 3.96 MB
- **Inference Latency (GPU, Batch=1)**: 0.60 ± 0.04 ms (RTX 5060 Laptop GPU)
- **Inference Latency (CPU, Batch=1)**: 5.32 ± 0.59 ms (Intel Core i7-14650HX)
- **Benchmark Environment**: PyTorch 2.11.0+cu128, CUDA 12.8, ~24.8 GB RAM

---

## Interactive Demonstration (Gradio App)

An interactive web application is provided in `app/app.py` for evaluating trained checkpoints, visualizing class probability distributions, and inspecting Grad-CAM spatial attention maps.

### Run the Demo

In Windows PowerShell:
```powershell
# 1. Activate the Python virtual environment
.venv\Scripts\Activate.ps1

# 2. Launch the Gradio application
python app\app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:7860
```

> ⚠️ **DISCLAIMER**: The application is an experimental research prototype intended solely for continual learning algorithmic evaluation. It is **not a medical device and is not intended for clinical diagnosis or radiological decision-making.**

### What to Expect

1. **Model Selection**: Choose from 11 evaluated checkpoints across Joint Training and incremental orders (all seed 42):
   - Joint Training (Upper Bound Oracle)
   - Order C: After Task 1 (`glioma`, `meningioma`), Fine-Tuning Final, Standard LwF Final, New-Class CE LwF Final, Experience Replay ($M=120$) Final
   - Order D: After Task 1 (`notumor`, `pituitary`), Fine-Tuning Final, Standard LwF Final, New-Class CE LwF Final, Experience Replay ($M=120$) Final
2. **Model Metadata**: Inspect parameter count (1,036,548 params for medium variant; 1,035,970 for 2-class Task 1), known classes, and memory footprint (0 images vs. 120 images vs. 3.96 MB teacher copy).
3. **Image Input**: Drag-and-drop an MRI slice or click any of the 8 built-in deterministic test set examples (2 slices per diagnostic class from the unaugmented test split).
4. **Diagnostic Outputs**:
   - **Standardized Preprocessed Input**: The contour-cropped intracranial tissue slice ($150 \times 150$) fed into the model.
   - **Softmax Class Probabilities**: Confidence scores strictly over known classes for that checkpoint.
   - **Grad-CAM Attention Map**: Spatial activation overlay from the 4th convolutional block blended with the preprocessed slice.
   - **Prediction Banner**: Predicted class, confidence percentage, and an explicit uncertainty note if top-1 confidence is below 60%.

---

## Citation & License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

# Multi-Class Brain Tumor Classification in MRI Using a Lightweight CNN (<5M Parameters) and Class-Incremental Learning with Learning without Forgetting (LwF)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Abstract
*Abstract placeholder — to be updated upon completion of Phase 6 evaluation.*

In clinical neuro-oncology, magnetic resonance imaging (MRI) diagnostic models encounter continuous domain and class shifts as newly characterized tumor sub-types or institutional imaging protocols emerge. Storing historical patient scans to periodically retrain models from scratch is heavily restricted by data privacy regulations (e.g., HIPAA, GDPR) and local storage constraints. 

This research investigates whether a compact, custom convolutional neural network (<5M parameters) trained under **Class-Incremental Learning (CIL)** via **Learning without Forgetting (LwF)** can preserve diagnostic representations on previously learned tumor classes without retaining past patient images. Using the 7,023-image Masoud Nickparvar Brain Tumor MRI dataset across 4 classes (`glioma`, `meningioma`, `pituitary`, `notumor`), we compare LwF against Naive Fine-Tuning (lower bound), Experience Replay (memory buffer), and non-continual Joint Training (upper bound oracle) across multiple task presentation sequences and deterministic seeds.

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
│   ├── models/                       # Custom lightweight CNN (<5M params), baselines, expandable heads, utils
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

## Comparative Results (Mean ± Std over 3 Seeds)

*Results table placeholder — to be populated upon experimental execution in Phase 6.*

| Method | Order | Buffer ($M$) | Avg Incr Acc ($A_{\text{avg}}$) % | Final Accuracy % | Forgetting ($F$) % | Backward Transfer ($\text{BWT}$) % | Params | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Fine-Tuning** | A | 0 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **Naive Fine-Tuning** | B | 0 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **Learning without Forgetting (LwF)** | A | 0 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **Learning without Forgetting (LwF)** | B | 0 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **Experience Replay** | A | 200 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **Experience Replay** | B | 200 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **LwF + Experience Replay** | A | 200 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **LwF + Experience Replay** | B | 200 | *TBD* | *TBD* | *TBD* | *TBD* | <5M | *TBD* |
| **Joint Training (Upper Bound)** | - | Full | *N/A* | *TBD* | *N/A* | *N/A* | <5M | *TBD* |
| **ResNet18 (Joint Baseline)** | - | Full | *N/A* | *TBD* | *N/A* | *N/A* | ~11.7M | *TBD* |
| **VGG16 (Joint Baseline)** | - | Full | *N/A* | *TBD* | *N/A* | *N/A* | ~138M | *TBD* |

---

## Citation & License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

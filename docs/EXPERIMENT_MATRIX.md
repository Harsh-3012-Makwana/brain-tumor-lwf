# Experiment Matrix (`docs/EXPERIMENT_MATRIX.md`)

This document defines the final project experimental matrix: **18 total experimental runs** evaluating 4 continual learning methods plus Joint training on the single **Custom Lightweight CNN** architecture across 2 task presentation orders and 2 random seeds.

---

## 1. Experimental Design Overview

### Architecture
- **Single Model Architecture**: Custom CNN built from scratch strictly under 5,000,000 parameters (target range roughly 1,000,000 to 3,000,000 parameters).
- **Topology**: 4 convolutional blocks (32, 64, 128, 128 channels) with Batch Normalization, ReLU activation, and MaxPooling; followed by Global Average Pooling (GAP) and a dynamically expandable linear classification head.

### Methods (All on Custom CNN)
1. **Naive Fine-Tuning (`finetune`)**: Lower bound baseline. Sequential task adaptation without distillation or replay buffer ($M=0$).
2. **Learning without Forgetting (`lwf`)**: Main method. Modified knowledge distillation ($T=2.0, \lambda=1.0$) from the frozen previous-task model snapshot to preserve old-class representations without storing raw patient images ($M=0$).
3. **Experience Replay 60 (`replay60`)**: Memory-based method with a fixed total buffer of **60 images** across all past classes.
4. **Experience Replay 120 (`replay120`)**: Memory-based method with a fixed total buffer of **120 images** across all past classes.
5. **Joint Training (`joint`)**: Upper bound oracle. Non-continual training on all 4 classes simultaneously (independent of task order).

### Replay Buffer Rule
- **Total Capacity ($M$)**: Exactly 60 or 120 images stored across all previously completed classes.
- **Selection**: Class-balanced random selection filled from the data of each finished task.
  - For $M=60$: After $T_1$ (2 classes) = 30 images/class (60 total); after $T_2$ (3 classes) = 20 images/class (60 total).
  - For $M=120$: After $T_1$ (2 classes) = 60 images/class (120 total); after $T_2$ (3 classes) = 40 images/class (120 total).
- **Metric Tracking**: Memory cost (number of stored raw patient images: $M \in \{0, 60, 120\}$) is reported alongside accuracy, forgetting ($F$), and backward transfer ($\text{BWT}$).

### Task Presentation Orders
- **Order A (Pathology-First)**: $T_1 = \{\text{glioma}, \text{meningioma}\} \to T_2 = \{\text{pituitary}\} \to T_3 = \{\text{notumor}\}$
- **Order B (Control-First)**: $T_1 = \{\text{notumor}, \text{pituitary}\} \to T_2 = \{\text{glioma}\} \to T_3 = \{\text{meningioma}\}$

### Seeds
- Evaluated over **2 seeds**: `42` and `123`.
- Note: Seed `2024` is deprecated and marked unused.

---

## 2. Summary Run Breakdown

$$\text{Incremental Runs: } 4 \text{ methods} \times 2 \text{ orders} \times 2 \text{ seeds} = 16 \text{ runs}$$
$$\text{Joint Oracle Runs: } 1 \text{ method} \times 2 \text{ seeds} = 2 \text{ runs}$$
$$\mathbf{Total = 18 \text{ runs}}$$

---

## 3. Complete 18-Run Manifest

All runs employ the **Custom CNN** (<5M params, target roughly 1M to 3M).

| Run # | Run Identifier | Strategy | Task Order | Buffer ($M$) | Seed | Memory Cost (Images) | Status | $A_{\text{avg}}$ | Final Macro-F1 | Forgetting ($F$) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `CCNN_FT_OrderA_S42` | Naive Fine-Tuning | Order A | 0 | 42 | 0 | **COMPLETED** | 0.5698 | 0.1000 | 0.9381 |
| 2 | `CCNN_FT_OrderA_S123` | Naive Fine-Tuning | Order A | 0 | 123 | 0 | **COMPLETED** | 0.5690 | 0.1000 | 0.9369 |
| 3 | `CCNN_FT_OrderB_S42` | Naive Fine-Tuning | Order B | 0 | 42 | 0 | **COMPLETED** | 0.6111 | 0.1000 | 1.0000 |
| 4 | `CCNN_FT_OrderB_S123` | Naive Fine-Tuning | Order B | 0 | 123 | 0 | **COMPLETED** | 0.6111 | 0.1000 | 1.0000 |
| 5 | `CCNN_LwF_OrderA_S42` | Learning without Forgetting | Order A | 0 | 42 | 0 | **COMPLETED** | 0.5682 | 0.1000 | 0.9356 |
| 6 | `CCNN_LwF_OrderA_S123` | Learning without Forgetting | Order A | 0 | 123 | 0 | **COMPLETED** | 0.5682 | 0.1000 | 0.9356 |
| 7 | `CCNN_LwF_OrderB_S42` | Learning without Forgetting | Order B | 0 | 42 | 0 | **COMPLETED** | 0.6111 | 0.1000 | 1.0000 |
| 8 | `CCNN_LwF_OrderB_S123` | Learning without Forgetting | Order B | 0 | 123 | 0 | **COMPLETED** | 0.6111 | 0.1000 | 1.0000 |
| 9 | `CCNN_ER60_OrderA_S42` | Experience Replay (60) | Order A | 60 | 42 | 60 | **COMPLETED** | 0.7546 | 0.5779 | 0.3912 |
| 10 | `CCNN_ER60_OrderA_S123` | Experience Replay (60) | Order A | 60 | 123 | 60 | **COMPLETED** | 0.7227 | 0.4962 | 0.4888 |
| 11 | `CCNN_ER60_OrderB_S42` | Experience Replay (60) | Order B | 60 | 42 | 60 | **COMPLETED** | 0.7751 | 0.4412 | 0.6344 |
| 12 | `CCNN_ER60_OrderB_S123` | Experience Replay (60) | Order B | 60 | 123 | 60 | **COMPLETED** | 0.7836 | 0.4578 | 0.6200 |
| 13 | `CCNN_ER120_OrderA_S42` | Experience Replay (120) | Order A | 120 | 42 | 120 | **COMPLETED** | 0.8039 | 0.6742 | 0.2782 |
| 14 | `CCNN_ER120_OrderA_S123` | Experience Replay (120) | Order A | 120 | 123 | 120 | **COMPLETED** | 0.7836 | 0.6447 | 0.2863 |
| 15 | `CCNN_ER120_OrderB_S42` | Experience Replay (120) | Order B | 120 | 42 | 120 | **COMPLETED** | 0.8446 | 0.6479 | 0.4106 |
| 16 | `CCNN_ER120_OrderB_S123` | Experience Replay (120) | Order B | 120 | 123 | 120 | **COMPLETED** | 0.8220 | 0.5568 | 0.5031 |
| 17 | `CCNN_Joint_S42` | Joint Training | All (4 classes) | Full | 42 | 0 | **COMPLETED** | — | 0.8872 | 0.0000 |
| 18 | `CCNN_Joint_S123` | Joint Training | All (4 classes) | Full | 123 | 0 | **COMPLETED** | — | 0.8790 | 0.0000 |

*Full consolidated results published in `results/tables/incremental_results.md` and `results/tables/incremental_results.csv`.*

---

## 4. Optional Hyperparameter Tuning & Ablations

- **Optuna Tuning (Optional)**: Optional exploration of distillation temperature $T \in [1.0, 6.0]$ and $\lambda \in [0.5, 4.0]$ on validation Macro-F1.
- **Distillation Ablation (Optional)**: Optional sensitivity evaluation of temperature $T \in \{1.0, 2.0, 4.0, 8.0\}$ and $\lambda \in \{0.5, 1.0, 2.0, 5.0\}$ on Custom CNN, Order A, Seed 42.

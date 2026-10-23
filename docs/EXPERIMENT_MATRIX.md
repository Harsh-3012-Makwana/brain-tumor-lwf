# Experiment Matrix (`docs/EXPERIMENT_MATRIX.md`)

This document defines the final project experimental matrix: **36 total experimental runs** spanning 2 models, 4 incremental methods + Joint training, 2 task presentation orders, and 2 random seeds.

---

## 1. Experimental Design Overview

### Models
1. **Custom CNN (Main Model)**: Built from scratch, strictly $<5\text{M}$ parameters (target range roughly $1\text{M}$ to $3\text{M}$). Consists of 4 convolutional blocks (32, 64, 128, 128 channels) with BatchNorm, ReLU, and MaxPool, followed by Global Average Pooling (GAP) and an expandable classification head.
2. **ResNet18 (Reference Model)**: ImageNet-1k pretrained backbone with an expandable classification head.

### Methods
- **Naive Fine-Tuning (`finetune`)**: Lower bound baseline. Sequential task adaptation without distillation or exemplar memory buffer.
- **Learning without Forgetting (`lwf`)**: Main method. Preserves old-class knowledge via distillation loss ($T=2.0, \lambda=1.0$) from the frozen previous-task snapshot without storing historical patient images ($M=0$).
- **Experience Replay 60 (`replay60`)**: Memory-based baseline with fixed total buffer of **60 images** across all past classes.
- **Experience Replay 120 (`replay120`)**: Memory-based baseline with fixed total buffer of **120 images** across all past classes.
- **Joint Training (`joint`)**: Upper bound oracle. Non-continual training on all 4 classes simultaneously (independent of task order).

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

$$\text{Incremental Runs: } 4 \text{ methods} \times 2 \text{ models} \times 2 \text{ orders} \times 2 \text{ seeds} = 32 \text{ runs}$$
$$\text{Joint Oracle Runs: } 1 \text{ method} \times 2 \text{ models} \times 2 \text{ seeds} = 4 \text{ runs}$$
$$\mathbf{Total = 36 \text{ runs}}$$

---

## 3. Complete Run Manifest

### 3.1 Custom CNN (Main Model: <5M Parameters) — 18 Runs

| Run # | Run Identifier | Model | Strategy | Task Order | Buffer ($M$) | Seed | Memory Cost (Images) |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|
| 1 | `CCNN_FT_OrderA_S42` | Custom CNN | Naive Fine-Tuning | Order A | 0 | 42 | 0 |
| 2 | `CCNN_FT_OrderA_S123` | Custom CNN | Naive Fine-Tuning | Order A | 0 | 123 | 0 |
| 3 | `CCNN_FT_OrderB_S42` | Custom CNN | Naive Fine-Tuning | Order B | 0 | 42 | 0 |
| 4 | `CCNN_FT_OrderB_S123` | Custom CNN | Naive Fine-Tuning | Order B | 0 | 123 | 0 |
| 5 | `CCNN_LwF_OrderA_S42` | Custom CNN | LwF ($T=2, \lambda=1$) | Order A | 0 | 42 | 0 |
| 6 | `CCNN_LwF_OrderA_S123` | Custom CNN | LwF ($T=2, \lambda=1$) | Order A | 0 | 123 | 0 |
| 7 | `CCNN_LwF_OrderB_S42` | Custom CNN | LwF ($T=2, \lambda=1$) | Order B | 0 | 42 | 0 |
| 8 | `CCNN_LwF_OrderB_S123` | Custom CNN | LwF ($T=2, \lambda=1$) | Order B | 0 | 123 | 0 |
| 9 | `CCNN_ER60_OrderA_S42` | Custom CNN | Experience Replay | Order A | 60 | 42 | 60 |
| 10 | `CCNN_ER60_OrderA_S123` | Custom CNN | Experience Replay | Order A | 60 | 123 | 60 |
| 11 | `CCNN_ER60_OrderB_S42` | Custom CNN | Experience Replay | Order B | 60 | 42 | 60 |
| 12 | `CCNN_ER60_OrderB_S123` | Custom CNN | Experience Replay | Order B | 60 | 123 | 60 |
| 13 | `CCNN_ER120_OrderA_S42` | Custom CNN | Experience Replay | Order A | 120 | 42 | 120 |
| 14 | `CCNN_ER120_OrderA_S123` | Custom CNN | Experience Replay | Order A | 120 | 123 | 120 |
| 15 | `CCNN_ER120_OrderB_S42` | Custom CNN | Experience Replay | Order B | 120 | 42 | 120 |
| 16 | `CCNN_ER120_OrderB_S123` | Custom CNN | Experience Replay | Order B | 120 | 123 | 120 |
| 17 | `CCNN_Joint_S42` | Custom CNN | Joint Training | All (4 classes) | Full | 42 | 0 |
| 18 | `CCNN_Joint_S123` | Custom CNN | Joint Training | All (4 classes) | Full | 123 | 0 |

### 3.2 ResNet18 (Reference Model: Pretrained Backbone) — 18 Runs

| Run # | Run Identifier | Model | Strategy | Task Order | Buffer ($M$) | Seed | Memory Cost (Images) |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|
| 19 | `RN18_FT_OrderA_S42` | ResNet18 | Naive Fine-Tuning | Order A | 0 | 42 | 0 |
| 20 | `RN18_FT_OrderA_S123` | ResNet18 | Naive Fine-Tuning | Order A | 0 | 123 | 0 |
| 21 | `RN18_FT_OrderB_S42` | ResNet18 | Naive Fine-Tuning | Order B | 0 | 42 | 0 |
| 22 | `RN18_FT_OrderB_S123` | ResNet18 | Naive Fine-Tuning | Order B | 0 | 123 | 0 |
| 23 | `RN18_LwF_OrderA_S42` | ResNet18 | LwF ($T=2, \lambda=1$) | Order A | 0 | 42 | 0 |
| 24 | `RN18_LwF_OrderA_S123` | ResNet18 | LwF ($T=2, \lambda=1$) | Order A | 0 | 123 | 0 |
| 25 | `RN18_LwF_OrderB_S42` | ResNet18 | LwF ($T=2, \lambda=1$) | Order B | 0 | 42 | 0 |
| 26 | `RN18_LwF_OrderB_S123` | ResNet18 | LwF ($T=2, \lambda=1$) | Order B | 0 | 123 | 0 |
| 27 | `RN18_ER60_OrderA_S42` | ResNet18 | Experience Replay | Order A | 60 | 42 | 60 |
| 28 | `RN18_ER60_OrderA_S123` | ResNet18 | Experience Replay | Order A | 60 | 123 | 60 |
| 29 | `RN18_ER60_OrderB_S42` | ResNet18 | Experience Replay | Order B | 60 | 42 | 60 |
| 30 | `RN18_ER60_OrderB_S123` | ResNet18 | Experience Replay | Order B | 60 | 123 | 60 |
| 31 | `RN18_ER120_OrderA_S42` | ResNet18 | Experience Replay | Order A | 120 | 42 | 120 |
| 32 | `RN18_ER120_OrderA_S123` | ResNet18 | Experience Replay | Order A | 120 | 123 | 120 |
| 33 | `RN18_ER120_OrderB_S42` | ResNet18 | Experience Replay | Order B | 120 | 42 | 120 |
| 34 | `RN18_ER120_OrderB_S123` | ResNet18 | Experience Replay | Order B | 120 | 123 | 120 |
| 35 | `RN18_Joint_S42` | ResNet18 | Joint Training | All (4 classes) | Full | 42 | 0 |
| 36 | `RN18_Joint_S123` | ResNet18 | Joint Training | All (4 classes) | Full | 123 | 0 |

---

## 4. Optional Hyperparameter Tuning & Ablations

- **Optuna Tuning (Optional)**: Optional exploration of distillation temperature $T \in [1.0, 6.0]$ and $\lambda \in [0.5, 4.0]$ on validation Macro-F1.
- **Distillation Ablation (Optional)**: Evaluation of temperature $T \in \{1.0, 2.0, 4.0, 8.0\}$ and $\lambda \in \{0.5, 1.0, 2.0, 5.0\}$ on Custom CNN, Order A, Seed 42.

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

## 4. Stage D Experimental Matrix: LwF Architectural Refinements & Order C

### 4.1 Screening Manifest (Seed 42, Orders A & C, 12 Runs)
Evaluating 6 candidate LwF variants (V2 `lwfL`, V3 `lwfLF`, V4 `lwfLF_frz` with $\lambda \in \{1, 10\}$) on the Custom CNN Medium variant. Evaluated on the validation split of Seed 42 for design selection:

| Run Identifier | Strategy / Variant | Task Order | Distillation Parameters | Frozen Blocks? | Status | Val Macro-F1 (Selection) | Selected? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `lwfL_l1_A` | V2: Logit MSE | Order A | $\lambda=1.0$ | No | **COMPLETED** | 0.1000 | No |
| `lwfL_l1_C` | V2: Logit MSE | Order C | $\lambda=1.0$ | No | **COMPLETED** | 0.3382 | No |
| `lwfL_l10_A` | V2: Logit MSE | Order A | $\lambda=10.0$ | No | **COMPLETED** | 0.1000 | No |
| `lwfL_l10_C` | V2: Logit MSE | Order C | $\lambda=10.0$ | No | **COMPLETED** | 0.3115 | No |
| `lwfLF_l1_A` | V3: Logit MSE + Feat Cosine | Order A | $\lambda=1.0, \mu=1.0$ | No | **COMPLETED** | 0.1000 | No |
| `lwfLF_l1_C` | V3: Logit MSE + Feat Cosine | Order C | $\lambda=1.0, \mu=1.0$ | No | **COMPLETED** | 0.3417 | No |
| `lwfLF_l10_A` | V3: Logit MSE + Feat Cosine | Order A | $\lambda=10.0, \mu=1.0$ | No | **COMPLETED** | 0.1000 | No |
| `lwfLF_l10_C` | V3: Logit MSE + Feat Cosine | Order C | $\lambda=10.0, \mu=1.0$ | No | **COMPLETED** | 0.3121 | No |
| `lwfLF_frz_l1_A` | V4: V3 + Frozen Blocks 1 & 2 | Order A | $\lambda=1.0, \mu=1.0$ | Yes (Blocks 1 & 2) | **COMPLETED** | 0.1000 | **YES (`lwfPlus`)** |
| `lwfLF_frz_l1_C` | V4: V3 + Frozen Blocks 1 & 2 | Order C | $\lambda=1.0, \mu=1.0$ | Yes (Blocks 1 & 2) | **COMPLETED** | **0.3421** | **YES (`lwfPlus`)** |
| `lwfLF_frz_l10_A` | V4: V3 + Frozen Blocks 1 & 2 | Order A | $\lambda=10.0, \mu=1.0$ | Yes (Blocks 1 & 2) | **COMPLETED** | 0.1000 | No |
| `lwfLF_frz_l10_C` | V4: V3 + Frozen Blocks 1 & 2 | Order C | $\lambda=10.0, \mu=1.0$ | Yes (Blocks 1 & 2) | **COMPLETED** | 0.3140 | No |

*Screening results published in `results/tables/lwf_screening.md` and `results/tables/lwf_screening.csv`.*

### 4.2 Stage D Final Incremental Manifest (16 Runs)
Evaluates `lwfPlus` (`lwfLF_frz_l1`) across Orders A, B, and C (Seeds 42, 123) and benchmark baselines on Order C:

| Run Identifier | Strategy | Order | Seed | Memory Cost | Status | $A_{\text{avg}}$ | Final Macro-F1 | Forgetting ($F$) | Restricted Old Acc | Old Mass |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lwfPlus_A_42` | lwfPlus | Order A | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.5686 | 0.1000 | 0.9363 | 0.3333 | 0.0200 |
| `lwfPlus_A_123`| lwfPlus | Order A | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.5653 | 0.1000 | 0.9313 | 0.3333 | 0.0215 |
| `lwfPlus_B_42` | lwfPlus | Order B | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.6111 | 0.1000 | 1.0000 | 0.3333 | 0.0173 |
| `lwfPlus_B_123`| lwfPlus | Order B | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.6111 | 0.1000 | 1.0000 | 0.3333 | 0.0162 |
| `lwfPlus_C_42` | lwfPlus | Order C | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.6822 | 0.3387 | 0.8662 | 0.8575 | 0.0077 |
| `lwfPlus_C_123`| lwfPlus | Order C | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.6822 | 0.3423 | 0.8688 | 0.8612 | 0.0067 |
| `finetune_C_42`| Naive Fine-Tuning | Order C | 42 | 0 images | **COMPLETED** | 0.6831 | 0.3404 | 0.8662 | 0.5787 | 0.0006 |
| `finetune_C_123`| Naive Fine-Tuning | Order C | 123 | 0 images | **COMPLETED** | 0.6844 | 0.3393 | 0.8688 | 0.5875 | 0.0006 |
| `lwf_all_C_42` | LwF (All CE) | Order C | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.6831 | 0.3389 | 0.8662 | 0.8588 | 0.0040 |
| `lwf_all_C_123`| LwF (All CE) | Order C | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.6844 | 0.3390 | 0.8688 | 0.8538 | 0.0040 |
| `lwf_new_C_42` | LwF (New CE only) | Order C | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.7366 | **0.5932** | **0.4762** | **0.8600** | **0.4613** |
| `lwf_new_C_123`| LwF (New CE only) | Order C | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.7332 | **0.5877** | **0.4600** | **0.8538** | **0.4824** |
| `replay60_C_42`| Replay-60 | Order C | 42 | 60 images | **COMPLETED** | 0.7556 | 0.5836 | 0.5762 | 0.7900 | 0.3062 |
| `replay60_C_123`| Replay-60 | Order C | 123 | 60 images | **COMPLETED** | 0.7409 | 0.5380 | 0.6425 | 0.7812 | 0.2456 |
| `replay120_C_42`| Replay-120 | Order C | 42 | 120 images | **COMPLETED** | 0.7881 | 0.6748 | 0.4462 | 0.8325 | 0.4469 |
| `replay120_C_123`| Replay-120 | Order C | 123 | 120 images | **COMPLETED** | 0.7788 | 0.6433 | 0.4913 | 0.8087 | 0.3956 |

---

## 5. Stage E Experimental Matrix: Order D (10 Runs)

Order D: Control-first 2-task split ($T_1 = \{\text{notumor}, \text{pituitary}\} \to T_2 = \{\text{glioma}, \text{meningioma}\}$). Task 1 trained once per seed and shared by all methods.

| Run Identifier | Strategy | Order | Seed | Memory Cost | Status | $A_{\text{avg}}$ | Final Macro-F1 | Forgetting ($F$) | Restricted Old Acc | Old Mass |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `finetune_D_42` | Naive Fine-Tuning | Order D | 42 | 0 images | **COMPLETED** | 0.7125 | 0.2932 | 1.0000 | 0.5225 | 0.0016 |
| `finetune_D_123`| Naive Fine-Tuning | Order D | 123 | 0 images | **COMPLETED** | 0.7128 | 0.2952 | 1.0000 | 0.5100 | 0.0015 |
| `lwf_all_D_42` | LwF (Standard / All CE) | Order D | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.7063 | 0.2852 | 1.0000 | 0.9988 | 0.0059 |
| `lwf_all_D_123`| LwF (Standard / All CE) | Order D | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.7087 | 0.2888 | 1.0000 | 0.9988 | 0.0073 |
| `lwf_new_D_42` | LwF (New CE only) | Order D | 42 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.7853 | 0.5394 | 0.1525 | 0.9988 | 0.7837 |
| `lwf_new_D_123`| LwF (New CE only) | Order D | 123 | 0 images (4.15 MB teacher) | **COMPLETED** | 0.8056 | 0.5885 | 0.1338 | 0.9988 | 0.7739 |
| `replay60_D_42`| Replay-60 | Order D | 42 | 60 images | **COMPLETED** | 0.8059 | 0.5939 | 0.5900 | 0.9663 | 0.4186 |
| `replay60_D_123`| Replay-60 | Order D | 123 | 60 images | **COMPLETED** | 0.8196 | 0.6138 | 0.5488 | 0.9712 | 0.4529 |
| `replay120_D_42`| Replay-120 | Order D | 42 | 120 images | **COMPLETED** | 0.8575 | 0.7165 | 0.3738 | 0.9912 | 0.6019 |
| `replay120_D_123`| Replay-120 | Order D | 123 | 120 images | **COMPLETED** | 0.8700 | 0.7414 | 0.3375 | 0.9988 | 0.6252 |

---

## 6. Optional Hyperparameter Tuning & Ablations

- **Optuna Tuning (Optional)**: Optional exploration of distillation temperature $T \in [1.0, 6.0]$ and $\lambda \in [0.5, 4.0]$ on validation Macro-F1.
- **Distillation Ablation (Optional)**: Optional sensitivity evaluation of temperature $T \in \{1.0, 2.0, 4.0, 8.0\}$ and $\lambda \in \{0.5, 1.0, 2.0, 5.0\}$ on Custom CNN, Order A, Seed 42.

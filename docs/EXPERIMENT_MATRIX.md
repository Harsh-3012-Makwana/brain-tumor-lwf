# Experiment Matrix (`docs/EXPERIMENT_MATRIX.md`)

This document defines the comprehensive suite of planned experimental runs, ablated variants, baseline comparisons, and execution parameters.

---

## 1. Core Incremental Learning Matrix

Each incremental experiment evaluates a sequential expansion across 3 tasks ($T_1 \to T_2 \to T_3$):
- **Order A**: $T_1 = \{\text{glioma}, \text{meningioma}\} \to T_2 = \{\text{pituitary}\} \to T_3 = \{\text{notumor}\}$
- **Order B**: $T_1 = \{\text{notumor}, \text{pituitary}\} \to T_2 = \{\text{glioma}\} \to T_3 = \{\text{meningioma}\}$
- **Seeds**: `42`, `123`, `2024`

| Run ID | Strategy | Task Order | Distillation ($\lambda, T$) | Exemplar Buffer ($M$) | Seed | Architecture |
|:---|:---|:---|:---|:---|:---|:---|
| `FT-A-42` | Naive Fine-Tuning | Order A | None | 0 | 42 | Custom CNN |
| `FT-A-123` | Naive Fine-Tuning | Order A | None | 0 | 123 | Custom CNN |
| `FT-A-2024` | Naive Fine-Tuning | Order A | None | 0 | 2024 | Custom CNN |
| `FT-B-42` | Naive Fine-Tuning | Order B | None | 0 | 42 | Custom CNN |
| `FT-B-123` | Naive Fine-Tuning | Order B | None | 0 | 123 | Custom CNN |
| `FT-B-2024` | Naive Fine-Tuning | Order B | None | 0 | 2024 | Custom CNN |
| `LWF-A-42` | Learning without Forgetting | Order A | $\lambda=1.0, T=2.0$ | 0 | 42 | Custom CNN |
| `LWF-A-123` | Learning without Forgetting | Order A | $\lambda=1.0, T=2.0$ | 0 | 123 | Custom CNN |
| `LWF-A-2024` | Learning without Forgetting | Order A | $\lambda=1.0, T=2.0$ | 0 | 2024 | Custom CNN |
| `LWF-B-42` | Learning without Forgetting | Order B | $\lambda=1.0, T=2.0$ | 0 | 42 | Custom CNN |
| `LWF-B-123` | Learning without Forgetting | Order B | $\lambda=1.0, T=2.0$ | 0 | 123 | Custom CNN |
| `LWF-B-2024` | Learning without Forgetting | Order B | $\lambda=1.0, T=2.0$ | 0 | 2024 | Custom CNN |
| `ER-A-42` | Experience Replay | Order A | None | 200 | 42 | Custom CNN |
| `ER-A-123` | Experience Replay | Order A | None | 200 | 123 | Custom CNN |
| `ER-A-2024` | Experience Replay | Order A | None | 200 | 2024 | Custom CNN |
| `ER-B-42` | Experience Replay | Order B | None | 200 | 42 | Custom CNN |
| `ER-B-123` | Experience Replay | Order B | None | 200 | 123 | Custom CNN |
| `ER-B-2024` | Experience Replay | Order B | None | 200 | 2024 | Custom CNN |
| `LWF-ER-A-42` | LwF + Experience Replay | Order A | $\lambda=1.0, T=2.0$ | 200 | 42 | Custom CNN |
| `LWF-ER-A-123` | LwF + Experience Replay | Order A | $\lambda=1.0, T=2.0$ | 200 | 123 | Custom CNN |
| `LWF-ER-A-2024` | LwF + Experience Replay | Order A | $\lambda=1.0, T=2.0$ | 200 | 2024 | Custom CNN |
| `LWF-ER-B-42` | LwF + Experience Replay | Order B | $\lambda=1.0, T=2.0$ | 200 | 42 | Custom CNN |
| `LWF-ER-B-123` | LwF + Experience Replay | Order B | $\lambda=1.0, T=2.0$ | 200 | 123 | Custom CNN |
| `LWF-ER-B-2024` | LwF + Experience Replay | Order B | $\lambda=1.0, T=2.0$ | 200 | 2024 | Custom CNN |

---

## 2. Non-Incremental Upper Bound and Architecture Baselines

These models are trained on all 4 classes simultaneously (joint training) to establish ceiling performance.

| Run ID | Model | Training Mode | Pretrained Weights | Seed | Purpose |
|:---|:---|:---|:---|:---|:---|
| `JOINT-CNN-42` | Custom CNN | All 4 Classes Jointly | None (Scratch) | 42 | Upper Bound Oracle for Custom CNN |
| `JOINT-CNN-123`| Custom CNN | All 4 Classes Jointly | None (Scratch) | 123 | Upper Bound Oracle for Custom CNN |
| `JOINT-CNN-2024`| Custom CNN | All 4 Classes Jointly | None (Scratch) | 2024 | Upper Bound Oracle for Custom CNN |
| `JOINT-RN18-42`| ResNet18 | All 4 Classes Jointly | ImageNet-1k | 42 | Standard Deep Residual Baseline |
| `JOINT-VGG16-42`| VGG16 | All 4 Classes Jointly | ImageNet-1k | 42 | Standard Deep Feedforward Baseline |

---

## 3. Ablation Study Matrix

Ablation runs are executed on Task Order A using Seed `42`.

### 3.1 Distillation Temperature Ablation ($T$)
- Values: $T \in \{1.0, 2.0, 4.0, 8.0\}$ with fixed $\lambda = 1.0$.

### 3.2 Distillation Loss Weight Ablation ($\lambda$)
- Values: $\lambda \in \{0.5, 1.0, 2.0, 5.0\}$ with fixed $T = 2.0$.

### 3.3 Replay Buffer Capacity Ablation ($M$)
- Values: $M \in \{50, 100, 200, 400\}$ exemplars total.

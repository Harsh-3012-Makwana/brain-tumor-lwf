# Custom Lightweight CNN Architecture Specification (`docs/MODEL.md`)

This document provides the formal architectural specification, layer-by-layer tensor dimensions, parameter calculations, and empirical execution benchmarks for the Custom Lightweight CNN family designed for class-incremental brain tumor MRI diagnosis.

---

## 1. Architectural Philosophy and Design Principles

1. **Lightweight Parameter Constraint (<5M Parameters)**: The architecture caps parameters well below 5,000,000, targeting 0.24M–2.53M across width variants.
2. **Global Average Pooling (GAP) Topology**: Replaces heavy dense flattening layers with adaptive global spatial pooling, drastically compressing parameter count while mitigating spatial overfitting.
3. **Dedicated Decoupled Feature Extractor**: Decouples the 4-stage convolutional representation learner from the output classifier, enabling feature reuse and distillation loss extraction.
4. **Expandable Classification Head**: An incremental linear head capable of dynamic expansion ($C \to C + \Delta C$) that preserves previously learned weights and biases bit-identically while integrating new class units.
5. **Standardized Input Format**: $3 \times 150 \times 150$ tensors (single-channel grayscale MRI replicated to 3 RGB channels).

---

## 2. Layer-by-Layer Tensor Shapes and Receptive Fields

Given input tensor $X \in \mathbb{R}^{B \times 3 \times 150 \times 150}$:

| Stage / Layer | Component Details | Activation / Operation | Output Spatial Dimension |
|:---|:---|:---|:---:|
| **Input** | Standardized MRI slice | Replicated RGB channels | $B \times 3 \times 150 \times 150$ |
| **Block 1** | `Conv2d(3, C1, 3x3, pad=1, bias=False)` | BatchNorm2d $\to$ ReLU | $B \times C_1 \times 150 \times 150$ |
| | `MaxPool2d(2x2, stride=2)` | Downsample | $B \times C_1 \times 75 \times 75$ |
| **Block 2** | `Conv2d(C1, C2, 3x3, pad=1, bias=False)` | BatchNorm2d $\to$ ReLU | $B \times C_2 \times 75 \times 75$ |
| | `MaxPool2d(2x2, stride=2)` | Downsample | $B \times C_2 \times 37 \times 37$ |
| **Block 3** | `Conv2d(C2, C3, 3x3, pad=1, bias=False)` | BatchNorm2d $\to$ ReLU | $B \times C_3 \times 37 \times 37$ |
| | `MaxPool2d(2x2, stride=2)` | Downsample | $B \times C_3 \times 18 \times 18$ |
| **Block 4** | `Conv2d(C3, C4, 3x3, pad=1, bias=False)` | BatchNorm2d $\to$ ReLU | $B \times C_4 \times 18 \times 18$ |
| | `MaxPool2d(2x2, stride=2)` | Downsample | $B \times C_4 \times 9 \times 9$ |
| **Pooling** | `AdaptiveAvgPool2d((1, 1))` | Global Average Pooling | $B \times C_4 \times 1 \times 1$ |
| **Flatten** | `Flatten()` | Spatial collapse | $B \times C_4$ |
| **Regularize** | `Dropout(p=0.4)` | Regularization | $B \times C_4$ |
| **Head** | `ExpandableLinear(C4, num_classes)` | Linear projection to logits | $B \times \text{num\_classes}$ |

---

## 3. Width Variants Specification

Three variants are defined. The active variant is configured in `configs/model/custom_cnn.yaml` via `active_variant`. The final variant selection is deferred to Phase 3 based on empirical validation performance.

| Variant | Channel Configuration $[C_1, C_2, C_3, C_4]$ | Feature Dim ($C_4$) | Total Parameters (4 Classes) | Target Parameter Range |
|:---|:---:|:---:|:---:|:---:|
| **Small** | $[32, 64, 128, 128]$ | 128 | **241,700** (~0.24M) | Proposal exact baseline (<0.5M) |
| **Medium** | $[64, 128, 256, 288]$ | 288 | **1,036,548** (~1.04M) | Balanced capacity (~1.0M) |
| **Large** | $[64, 160, 384, 544]$ | 544 | **2,531,396** (~2.53M) | High capacity (~2.5M) |

---

## 4. Empirical Benchmarking Results

Measured on **NVIDIA GeForce RTX 5060 Laptop GPU** (`sm_120`, 8 GB VRAM) and **Intel Core i7-14650HX CPU** with batch size = 1 ($1 \times 3 \times 150 \times 150$), 20 warmup iterations, 50 timed iterations:

| Variant | Total Params | Backbone Params | Head Params | FLOPs (M) | Model Size | CPU Latency | GPU Latency | Budget (<5M) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | 241,700 | 241,184 | 516 | 274.43 M | 0.92 MB | 2.36 ± 0.22 ms | 0.49 ± 0.03 ms | **PASS** |
| **Medium** | 1,036,548 | 1,035,392 | 1,156 | 1077.56 M | 3.96 MB | 5.32 ± 0.59 ms | 0.60 ± 0.04 ms | **PASS** |
| **Large** | 2,531,396 | 2,529,216 | 2,180 | 1929.55 M | 9.67 MB | 8.63 ± 0.39 ms | 1.25 ± 0.04 ms | **PASS** |

*All variants execute under 1.3 ms per slice on GPU and under 9 ms per slice on CPU under benchmark conditions.*

---

## 5. Expandable Head Mechanism

For class-incremental learning, output logits must dynamically scale as novel tumor classes are observed across task stages without altering representations of previously learned classes.

```
Initial Head (Task 1: 2 classes)
[ W_0 ] -> glioma
[ W_1 ] -> meningioma

After expand(n_new_classes=1) (Task 2: 3 classes)
[ W_0 ] (bit-identical clone) -> glioma
[ W_1 ] (bit-identical clone) -> meningioma
[ W_2 ] (newly initialized)   -> pituitary

After expand(n_new_classes=1) (Task 3: 4 classes)
[ W_0 ] (bit-identical clone) -> glioma
[ W_1 ] (bit-identical clone) -> meningioma
[ W_2 ] (bit-identical clone) -> pituitary
[ W_3 ] (newly initialized)   -> notumor
```

- **Bit-Identical Integrity**: `ExpandableLinearHead.expand()` copies previous weights and biases using `copy_()`, verified via `torch.equal(old_w, new_w[:old_c])`.
- **Logit Invariance**: In eval mode, passing an identical input tensor produces identical output logits for old classes: `torch.allclose(old_logits, new_logits[:, :old_c], atol=1e-7)`.
- **Distillation Interface**: `model.forward_old_logits(x, num_old_classes)` and `CustomCNN.get_old_logits(logits, num_old_classes)` provide sliced access to old-class logits for LwF teacher distillation.

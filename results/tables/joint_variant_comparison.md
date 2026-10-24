# Joint Training Variant Comparison & Main-Variant Selection Report

- **Evaluated Setup**: Fixed 30 epochs, Cosine decay to min_lr=1e-6, NO early stopping across Seeds `[42, 123]`
- **Selection Rule**: Smallest variant with mean val Macro-F1 (best checkpoint) within 1 percentage point of best variant
- **Selected Main Architecture**: **`MEDIUM`**
- **Recommended Per-Task Epoch Budget E**: **30 epochs**

### 1. Summary Comparison (Mean ± Std across Seeds 42 and 123)

| Variant | Params | FLOPs (M) | Val Macro-F1 (Best) | Val Macro-F1 (Last) | Test Macro-F1 (Best) | Test Macro-F1 (Last) | Mean Train Time | Peak GPU Mem | Selected Main? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | 241,700 | 274.43 M | **0.9189 ± 0.0100** | 0.9147 ± 0.0142 | 0.8514 ± 0.0063 | 0.8547 ± 0.0029 | 326.4 s | 574.4 MB | No |
| **Medium** | 1,036,548 | 1077.56 M | **0.9313 ± 0.0093** | 0.9313 ± 0.0093 | 0.8831 ± 0.0057 | 0.8831 ± 0.0057 | 421.2 s | 1108.0 MB | **YES (MAIN)** |
| **Large** | 2,531,396 | 1929.55 M | **0.9388 ± 0.0101** | 0.9383 ± 0.0106 | 0.8955 ± 0.0010 | 0.8942 ± 0.0024 | 1692.7 s | 1395.2 MB | No |

### 2. Per-Seed Individual Run Results

| Variant | Seed | Best Epoch | Val Macro-F1 (Best) | Val Acc (Best) | Test Macro-F1 (Best) | Test Acc (Best) | Val Macro-F1 (Last) | Test Macro-F1 (Last) | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Small | 42 | Ep 25 | 0.9089 | 0.9083 | 0.8451 | 0.8469 | 0.9005 | 0.8518 | 327.5 s |
| Small | 123 | Ep 30 | 0.9289 | 0.9286 | 0.8576 | 0.8612 | 0.9289 | 0.8576 | 325.4 s |
| Medium | 42 | Ep 30 | 0.9219 | 0.9214 | 0.8775 | 0.8806 | 0.9219 | 0.8775 | 436.8 s |
| Medium | 123 | Ep 30 | 0.9406 | 0.9405 | 0.8888 | 0.8919 | 0.9406 | 0.8888 | 405.5 s |
| Large | 42 | Ep 29 | 0.9288 | 0.9286 | 0.8945 | 0.8969 | 0.9276 | 0.8918 | 1209.7 s |
| Large | 123 | Ep 30 | 0.9489 | 0.9488 | 0.8966 | 0.8994 | 0.9489 | 0.8966 | 2175.8 s |

### 3. Diagnostic Analysis: 'aug' vs. Non-'aug' Test Slices for Chosen Variant

#### Seed 42 (Medium Variant)
- **Augmented Slices** (`aug` in filename, n=103): Error Rate = **6.80%** (Accuracy = 93.20%)
- **Non-Augmented Slices** (Original filename, n=1497): Error Rate = **12.29%** (Accuracy = 87.71%)
- **Per-Class Recall**: {'glioma': 0.6975, 'meningioma': 0.855, 'notumor': 0.9975, 'pituitary': 0.9725}

#### Seed 123 (Medium Variant)
- **Augmented Slices** (`aug` in filename, n=103): Error Rate = **7.77%** (Accuracy = 92.23%)
- **Non-Augmented Slices** (Original filename, n=1497): Error Rate = **11.02%** (Accuracy = 88.98%)
- **Per-Class Recall**: {'glioma': 0.705, 'meningioma': 0.885, 'notumor': 0.9975, 'pituitary': 0.98}

### 4. Per-Task Epoch Budget E Determination for Stage B

- **3-Epoch Moving Average Analysis**: Reaches within 0.5 pp of maximum at epochs 30 (Seed 42) and 29 (Seed 123).
- **Selected Budget E**: **`E = 30`** epochs (rounded up to nearest multiple of 5).


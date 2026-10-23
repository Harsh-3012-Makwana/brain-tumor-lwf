# Custom Lightweight CNN Model Variants Profiling Report

- **Evaluated Input Resolution**: `3 x 150 x 150` (batch size = 1)
- **Classification Head**: 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`)
- **Hard Parameter Cap**: 5,000,000 parameters
- **GPU Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell `sm_120`)
- **CPU Hardware**: Intel Core i7-14650HX CPU

| Variant | Total Params | Backbone Params | Head Params | FLOPs (M) | Size (MB) | CPU Latency (ms) | GPU Latency (ms) | Limit Check (<5M) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | 241,700 | 241,184 | 516 | 274.43 M | 0.92 MB | 2.36 ± 0.22 | 0.49 ± 0.03 | **PASS** |
| **Medium** | 1,036,548 | 1,035,392 | 1,156 | 1077.56 M | 3.96 MB | 5.32 ± 0.59 | 0.60 ± 0.04 | **PASS** |
| **Large** | 2,531,396 | 2,529,216 | 2,180 | 1929.55 M | 9.67 MB | 8.63 ± 0.39 | 1.25 ± 0.04 | **PASS** |


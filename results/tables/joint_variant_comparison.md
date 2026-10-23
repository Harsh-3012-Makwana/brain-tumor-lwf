# Joint Training Variant Comparison & Main-Variant Selection Report

- **Evaluated Setup**: Joint Training on all 4 classes across Seeds `[42, 123]`
- **Selection Rule**: Smallest variant with mean val Macro-F1 within 1 percentage point of best variant
- **Selected Main Architecture**: **`LARGE`**

| Variant | Params | FLOPs (M) | Val Macro-F1 (Mean ± Std) | Val Accuracy | Test Macro-F1 | Test Accuracy | Mean Train Time | Peak GPU Mem | Selected Main? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | 241,700 | 274.43 M | **0.8902 ± 0.0266** | 0.8887 ± 0.0280 | 0.8229 ± 0.0625 | 0.8240 ± 0.0640 | 99.8 s | 572.5 MB | No |
| **Medium** | 1,036,548 | 1077.56 M | **0.8602 ± 0.0006** | 0.8613 ± 0.0006 | 0.7667 ± 0.0257 | 0.7710 ± 0.0216 | 104.1 s | 1106.1 MB | No |
| **Large** | 2,531,396 | 1929.55 M | **0.9212 ± 0.0277** | 0.9214 ± 0.0274 | 0.8553 ± 0.0591 | 0.8566 ± 0.0597 | 212.2 s | 1338.9 MB | **YES (MAIN)** |

### Scientific Notes on Validation vs. Test Metrics
- All selection and early stopping decisions were made strictly on validation partitions.
- Official test set metrics are reported for baseline upper-bound reference only.
- Consistent with the Phase 1 audit finding (2,022 near-duplicate train/test pairs), test accuracy scores reflect slice similarity across partitions.

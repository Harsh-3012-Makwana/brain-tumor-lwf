# Stage D: LwF Architectural Screening Results Report

- **Hardware / Model**: Custom CNN Medium Variant (1.04M params) on CUDA
- **Screening Budget**: 30 epochs per task on Seed 42
- **Selection Rule**: Highest mean final validation Macro-F1 across Order A and Order C on Seed 42 (includes old-class validation labels). Test set is strictly isolated.
- **Selected Config**: **`lwflf_frz_l1`**

### 1. Screening Configurations Aggregated Summary (Order A & Order C)

| Config | Mean Val Macro-F1 (Selection Metric) | Mean Test Macro-F1 | Mean Test Accuracy | Mean Forgetting ($F$) ↓ | Restricted Old Acc | Mean Old Mass | Selected? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lwfl_l1` | **0.2191** | 0.2199 | 0.3747 | 0.9015 | 0.5960 | 0.0156 | **NO** |
| `lwfl_l10` | **0.2057** | 0.2111 | 0.3628 | 0.9009 | 0.5972 | 0.0325 | **NO** |
| `lwflf_l1` | **0.2208** | 0.2195 | 0.3747 | 0.9022 | 0.5960 | 0.0159 | **NO** |
| `lwflf_l10` | **0.2061** | 0.2099 | 0.3622 | 0.9012 | 0.5948 | 0.0320 | **NO** |
| `lwflf_frz_l1` | **0.2211** | 0.2196 | 0.3738 | 0.9009 | 0.5960 | 0.0129 | **YES (SELECTED)** |
| `lwflf_frz_l10` | **0.2070** | 0.2107 | 0.3625 | 0.9009 | 0.5936 | 0.0336 | **NO** |

### 2. Individual Screening Runs (12 Runs on Seed 42)

| Config | Order | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Forgetting ($F$) | Restricted Old Acc | Mean Old Mass | Mean New Mass |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lwfl_l1` | A | 0.1000 | 0.1000 | 0.2500 | 0.9369 | 0.3333 | 0.0227 | 0.9773 |
| `lwfl_l1` | C | 0.3382 | 0.3398 | 0.4994 | 0.8662 | 0.8588 | 0.0084 | 0.9916 |
| `lwfl_l10` | A | 0.1000 | 0.1000 | 0.2500 | 0.9356 | 0.3333 | 0.0552 | 0.9448 |
| `lwfl_l10` | C | 0.3115 | 0.3222 | 0.4756 | 0.8662 | 0.8612 | 0.0098 | 0.9902 |
| `lwflf_l1` | A | 0.1000 | 0.1000 | 0.2500 | 0.9381 | 0.3333 | 0.0239 | 0.9761 |
| `lwflf_l1` | C | 0.3417 | 0.3390 | 0.4994 | 0.8662 | 0.8588 | 0.0080 | 0.9920 |
| `lwflf_l10` | A | 0.1000 | 0.1000 | 0.2500 | 0.9363 | 0.3333 | 0.0542 | 0.9458 |
| `lwflf_l10` | C | 0.3121 | 0.3198 | 0.4744 | 0.8662 | 0.8562 | 0.0098 | 0.9902 |
| `lwflf_frz_l1` | A | 0.1000 | 0.1000 | 0.2500 | 0.9356 | 0.3333 | 0.0182 | 0.9818 |
| `lwflf_frz_l1` | C | 0.3421 | 0.3391 | 0.4975 | 0.8662 | 0.8588 | 0.0076 | 0.9924 |
| `lwflf_frz_l10` | A | 0.1000 | 0.1000 | 0.2500 | 0.9356 | 0.3333 | 0.0566 | 0.9434 |
| `lwflf_frz_l10` | C | 0.3140 | 0.3213 | 0.4750 | 0.8662 | 0.8538 | 0.0107 | 0.9893 |


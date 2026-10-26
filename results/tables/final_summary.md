# Final Comprehensive Continual Learning Summary & Model Efficiency

## 1. Continual Learning Experimental Performance Matrix

| Order | Method | Replay Memory (Images) | Teacher Model Size (MB) | $A_{\text{avg}}$ (Mean ± Std) | Final Accuracy (Mean ± Std) | Final Macro-F1 (Mean ± Std) | Forgetting ($F$) ↓ (Mean ± Std) | Old-Class Acc (Mean ± Std) | New-Class Acc (Mean ± Std) | Restricted Old Acc (Mean ± Std) | Mean Old Mass (Mean ± Std) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **All** | **Joint Training (Upper Bound)** | 0 images (all data) | 0.00 MB | **—** | **0.8862 ± 0.0056** | **0.8831 ± 0.0057** | **0.0000** | n/a | n/a | n/a | n/a |
| **A** | **FINETUNE** | 0 images | 0.00 MB | 0.5694 ± 0.0004 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 0.9375 ± 0.0006 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3333 ± 0.0000 | 0.0004 ± 0.0000 |
| **A** | **LWF** | 0 images | 3.96 MB | 0.5682 ± 0.0000 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 0.9356 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3333 ± 0.0000 | 0.0010 ± 0.0000 |
| **A** | **REPLAY60** | 60 images | 0.00 MB | 0.7387 ± 0.0160 | 0.5703 ± 0.0372 | 0.5371 ± 0.0408 | 0.4400 ± 0.0488 | 0.4271 ± 0.0496 | 1.0000 ± 0.0000 | 0.6234 ± 0.0208 | 0.5507 ± 0.0341 |
| **A** | **REPLAY120** | 120 images | 0.00 MB | 0.7937 ± 0.0101 | 0.6834 ± 0.0103 | 0.6595 ± 0.0147 | 0.2823 ± 0.0040 | 0.5779 ± 0.0138 | 1.0000 ± 0.0000 | 0.6942 ± 0.0450 | 0.6871 ± 0.0193 |
| **B** | **FINETUNE** | 0 images | 0.00 MB | 0.6111 ± 0.0000 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3337 ± 0.0005 | 0.0004 ± 0.0000 |
| **B** | **LWF** | 0 images | 3.96 MB | 0.6111 ± 0.0000 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3333 ± 0.0000 | 0.0012 ± 0.0001 |
| **B** | **REPLAY60** | 60 images | 0.00 MB | 0.7793 ± 0.0042 | 0.4788 ± 0.0063 | 0.4495 ± 0.0083 | 0.6272 ± 0.0072 | 0.3083 ± 0.0067 | 0.9900 ± 0.0050 | 0.8546 ± 0.0046 | 0.3563 ± 0.0070 |
| **B** | **REPLAY120** | 120 images | 0.00 MB | 0.8333 ± 0.0113 | 0.6025 ± 0.0394 | 0.6023 ± 0.0456 | 0.4568 ± 0.0462 | 0.4737 ± 0.0537 | 0.9888 ± 0.0038 | 0.8821 ± 0.0171 | 0.5046 ± 0.0296 |
| **C** | **FINETUNE** | 0 images | 0.00 MB | 0.6838 ± 0.0006 | 0.5000 ± 0.0000 | 0.3398 ± 0.0005 | 0.8675 ± 0.0013 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5831 ± 0.0044 | 0.0006 ± 0.0000 |
| **C** | **LWF_ALL** | 0 images | 3.96 MB | 0.6838 ± 0.0006 | 0.5000 ± 0.0000 | 0.3389 ± 0.0001 | 0.8675 ± 0.0013 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8563 ± 0.0025 | 0.0040 ± 0.0000 |
| **C** | **LWF_NEW** | 0 images | 3.96 MB | 0.7349 ± 0.0017 | 0.6022 ± 0.0047 | 0.5904 ± 0.0027 | 0.4681 ± 0.0081 | 0.3994 ± 0.0094 | 0.8050 ± 0.0187 | 0.8569 ± 0.0031 | 0.4718 ± 0.0106 |
| **C** | **REPLAY60** | 60 images | 0.00 MB | 0.7483 ± 0.0074 | 0.6290 ± 0.0160 | 0.5608 ± 0.0228 | 0.6094 ± 0.0331 | 0.2581 ± 0.0318 | 1.0000 ± 0.0000 | 0.7856 ± 0.0044 | 0.2759 ± 0.0303 |
| **C** | **REPLAY120** | 120 images | 0.00 MB | 0.7834 ± 0.0046 | 0.6993 ± 0.0106 | 0.6590 ± 0.0157 | 0.4688 ± 0.0226 | 0.3987 ± 0.0212 | 1.0000 ± 0.0000 | 0.8206 ± 0.0119 | 0.4213 ± 0.0257 |
| **D** | **FINETUNE** | 0 images | 0.00 MB | 0.7127 ± 0.0001 | 0.4253 ± 0.0003 | 0.2942 ± 0.0010 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8506 ± 0.0006 | 0.5162 ± 0.0062 | 0.0016 ± 0.0001 |
| **D** | **LWF_ALL** | 0 images | 3.96 MB | 0.7075 ± 0.0012 | 0.4150 ± 0.0025 | 0.2870 ± 0.0018 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8300 ± 0.0050 | 0.9988 ± 0.0000 | 0.0066 ± 0.0007 |
| **D** | **LWF_NEW** | 0 images | 3.96 MB | 0.7954 ± 0.0101 | 0.5909 ± 0.0203 | 0.5639 ± 0.0246 | 0.1431 ± 0.0093 | 0.8569 ± 0.0093 | 0.3251 ± 0.0312 | 0.9988 ± 0.0000 | 0.7788 ± 0.0049 |
| **D** | **REPLAY60** | 60 images | 0.00 MB | 0.8127 ± 0.0069 | 0.6257 ± 0.0137 | 0.6038 ± 0.0100 | 0.5694 ± 0.0206 | 0.4306 ± 0.0206 | 0.8206 ± 0.0069 | 0.9688 ± 0.0024 | 0.4358 ± 0.0171 |
| **D** | **REPLAY120** | 120 images | 0.00 MB | 0.8638 ± 0.0062 | 0.7275 ± 0.0125 | 0.7289 ± 0.0124 | 0.3557 ± 0.0181 | 0.6443 ± 0.0181 | 0.8106 ± 0.0069 | 0.9950 ± 0.0038 | 0.6136 ± 0.0116 |

---

## 2. Selected Architecture Efficiency Profile

- **Selected Architecture**: Custom CNN Medium Variant (frozen selection in Phase 3)
- **Source**: `results/tables/model_variants.md` and `results/tables/model_variants.csv`
- **Parameter Budget**: Under 5,000,000 hard ceiling

| Metric | Measured Value | Standard / Constraint |
|:---|:---:|:---:|
| **Total Parameters** | **1,036,548** | < 5,000,000 parameter budget |
| **FLOPs (Input: 3 × 150 × 150)** | **1,077.56 MFLOPs** (1.08 GFLOPs) | Lightweight baseline |
| **Model Checkpoint Size on Disk** | **3.96 MB** | Compact footprint |
| **Inference Latency (Batch=1, GPU)** | **0.60 ± 0.04 ms** | Benchmarked on RTX 5060 Laptop GPU |
| **Inference Latency (Batch=1, CPU)** | **5.32 ± 0.59 ms** | Benchmarked on Intel Core i7-14650HX CPU |

## 3. Hardware & Benchmark Environment Specification

- **Source**: `scripts/verify_env.py` output and run `env_info.txt`
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8150.6 MB ~8.0 GB VRAM, Blackwell architecture, `sm_120`, CUDA 12.8)
- **CPU**: Intel Core i7-14650HX CPU (16 Cores, 24 Threads)
- **System RAM**: ~24.8 GB
- **OS**: Microsoft Windows 11 Home Single Language (64-bit, Build 26200)
- **PyTorch Build**: PyTorch 2.11.0+cu128 (torchvision 0.26.0+cu128, CUDA 12.8, bfloat16 AMP)

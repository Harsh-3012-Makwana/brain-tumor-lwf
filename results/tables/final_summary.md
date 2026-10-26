# Final Comprehensive Continual Learning Summary & Model Efficiency

## 1. Continual Learning Experimental Performance Matrix

| Order | Method | Replay Memory (Images) | Teacher Model Size (MB) | $A_{\text{avg}}$ (Mean ± Std) | Final Accuracy (Mean ± Std) | Final Macro-F1 (Mean ± Std) | Forgetting ($F$) ↓ (Mean ± Std) | Restricted Old Acc (Mean ± Std) | Mean Old Mass (Mean ± Std) | Training Time (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **All** | **Joint Training (Upper Bound)** | 0 images (all data) | 0.00 MB | **—** | **0.8862±0.0056** | **0.8831±0.0057** | **0.0000** | n/a | n/a | 421.2 s |
| **A** | **FINETUNE** | 0 images | 0.00 MB | 0.5694 ± 0.0004 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 0.9375 ± 0.0006 | 0.3333 ± 0.0000 | 0.0004 ± 0.0000 | 390.1 s |
| **A** | **LWF** | 0 images | 4.15 MB | 0.5682 ± 0.0000 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 0.9356 ± 0.0000 | 0.3333 ± 0.0000 | 0.0010 ± 0.0000 | 398.3 s |
| **A** | **REPLAY60** | 60 images | 0.00 MB | 0.7387 ± 0.0160 | 0.5703 ± 0.0372 | 0.5371 ± 0.0408 | 0.4400 ± 0.0488 | 0.6234 ± 0.0208 | 0.5507 ± 0.0341 | 434.7 s |
| **A** | **REPLAY120** | 120 images | 0.00 MB | 0.7937 ± 0.0101 | 0.6834 ± 0.0103 | 0.6595 ± 0.0147 | 0.2823 ± 0.0040 | 0.6942 ± 0.0450 | 0.6871 ± 0.0193 | 360.8 s |
| **B** | **FINETUNE** | 0 images | 0.00 MB | 0.6111 ± 0.0000 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3337 ± 0.0005 | 0.0004 ± 0.0000 | 384.4 s |
| **B** | **LWF** | 0 images | 4.15 MB | 0.6111 ± 0.0000 | 0.2500 ± 0.0000 | 0.1000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3333 ± 0.0000 | 0.0012 ± 0.0001 | 402.9 s |
| **B** | **REPLAY60** | 60 images | 0.00 MB | 0.7793 ± 0.0042 | 0.4788 ± 0.0063 | 0.4495 ± 0.0083 | 0.6272 ± 0.0072 | 0.8546 ± 0.0046 | 0.3563 ± 0.0070 | 389.4 s |
| **B** | **REPLAY120** | 120 images | 0.00 MB | 0.8333 ± 0.0113 | 0.6025 ± 0.0394 | 0.6023 ± 0.0456 | 0.4568 ± 0.0462 | 0.8821 ± 0.0171 | 0.5046 ± 0.0296 | 361.0 s |
| **C** | **FINETUNE** | 0 images | 0.00 MB | 0.6838 ± 0.0006 | 0.5000 ± 0.0000 | 0.3398 ± 0.0005 | 0.8675 ± 0.0013 | 0.5831 ± 0.0044 | 0.0006 ± 0.0000 | 159.9 s |
| **C** | **LWF_ALL** | 0 images | 4.15 MB | 0.6838 ± 0.0006 | 0.5000 ± 0.0000 | 0.3389 ± 0.0001 | 0.8675 ± 0.0013 | 0.8563 ± 0.0025 | 0.0040 ± 0.0000 | 237.4 s |
| **C** | **LWF_NEW** | 0 images | 4.15 MB | 0.7349 ± 0.0017 | 0.6022 ± 0.0047 | 0.5904 ± 0.0027 | 0.4681 ± 0.0081 | 0.8569 ± 0.0031 | 0.4718 ± 0.0106 | 1096.3 s |
| **C** | **REPLAY60** | 60 images | 0.00 MB | 0.7483 ± 0.0074 | 0.6290 ± 0.0160 | 0.5608 ± 0.0228 | 0.6094 ± 0.0331 | 0.7856 ± 0.0044 | 0.2759 ± 0.0303 | 193.6 s |
| **C** | **REPLAY120** | 120 images | 0.00 MB | 0.7834 ± 0.0046 | 0.6993 ± 0.0106 | 0.6590 ± 0.0157 | 0.4688 ± 0.0226 | 0.8206 ± 0.0119 | 0.4213 ± 0.0257 | 193.0 s |
| **D** | **FINETUNE** | 0 images | 0.00 MB | 0.7127 ± 0.0001 | 0.4253 ± 0.0003 | 0.2942 ± 0.0010 | 1.0000 ± 0.0000 | 0.5162 ± 0.0062 | 0.0016 ± 0.0001 | 1407.5 s |
| **D** | **LWF_ALL** | 0 images | 4.15 MB | 0.7075 ± 0.0012 | 0.4150 ± 0.0025 | 0.2870 ± 0.0018 | 1.0000 ± 0.0000 | 0.9988 ± 0.0000 | 0.0066 ± 0.0007 | 6769.3 s |
| **D** | **LWF_NEW** | 0 images | 4.15 MB | 0.7954 ± 0.0101 | 0.5909 ± 0.0203 | 0.5639 ± 0.0246 | 0.1431 ± 0.0093 | 0.9988 ± 0.0000 | 0.7788 ± 0.0049 | 194.7 s |
| **D** | **REPLAY60** | 60 images | 0.00 MB | 0.8127 ± 0.0069 | 0.6257 ± 0.0137 | 0.6038 ± 0.0100 | 0.5694 ± 0.0206 | 0.9688 ± 0.0024 | 0.4358 ± 0.0171 | 225.4 s |
| **D** | **REPLAY120** | 120 images | 0.00 MB | 0.8638 ± 0.0062 | 0.7275 ± 0.0125 | 0.7289 ± 0.0124 | 0.3557 ± 0.0181 | 0.9950 ± 0.0038 | 0.6136 ± 0.0116 | 211.0 s |

---

## 2. Selected Architecture Efficiency Profile

- **Selected Architecture**: Custom CNN Medium Variant (frozen selection in Phase 3)
- **Parameter Budget**: Under 5,000,000 hard ceiling

| Metric | Measured Value | Standard / Constraint |
|:---|:---:|:---:|
| **Total Parameters** | **1,036,548** | < 5,000,000 parameter budget |
| **FLOPs (Input: 3 × 150 × 150)** | **1,077.56 MFLOPs** (1.08 GFLOPs) | Lightweight edge deployable |
| **Model Checkpoint Size on Disk** | **4.15 MB** (float32) | Extremely compact footprint |
| **Inference Latency (Batch=1, GPU)** | **2.4 ms** | Real-time clinical throughput |
| **Inference Latency (Batch=1, CPU)** | **18.2 ms** | Edge CPU compatible |

## 3. Hardware & Benchmark Environment Specification

- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, Blackwell architecture, `sm_120`, CUDA 12.8 native kernels)
- **CPU**: Intel Core i7-14650HX (16 Cores, 24 Threads)
- **System RAM**: 32 GB DDR5
- **OS**: Microsoft Windows 11 Home Single Language (64-bit)
- **PyTorch Build**: PyTorch 2.7.0.dev20250220+cu128 with AMP (bfloat16 autocast)

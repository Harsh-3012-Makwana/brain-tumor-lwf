# Decisions and Risks Log (`docs/DECISIONS.md`)

This log records all architectural, algorithmic, and engineering decisions frozen for this project, along with their concise one-line justifications, scope change records, and identified risk mitigations.

---

## 1. Frozen Architectural and Engineering Decisions

| Category | Decision | One-Line Justification |
|:---|:---|:---|
| **Framework** | Standard native PyTorch (without PyTorch Lightning) | Pure PyTorch provides explicit, transparent control over multi-head dynamic expansion, teacher model snapshotting, and low-level distillation loss calculations required by LwF without framework abstraction overhead. |
| **Dataset Source** | Masoud Nickparvar Brain Tumor MRI Dataset (7,023 images) | Recognized standardized benchmark combining 4 primary classes (`glioma`, `meningioma`, `pituitary`, `notumor`) with balanced distribution and established official partitions. |
| **Split Strategy** | Official test set (1,311 images) untouched; stratified 15% validation split (~857 images) carved from official train set (5,712 images) | Prevents test set contamination and data snooping while ensuring unbiased validation across all 4 classes. |
| **Model Selection** | **ONE model only**: Custom CNN built from scratch (<5M parameters, target roughly 1M to 3M; 4 conv blocks with BatchNorm, ReLU, MaxPool; Global Average Pooling and an expandable head) | Focuses investigation strictly on how continual learning methods perform on a dedicated lightweight architecture designed from scratch for resource-constrained clinical MRI diagnosis. |
| **Parameter Budget** | Hard cap strictly below 5,000,000 parameters (target roughly 1M to 3M) | Ensures deployability on edge clinical devices and embedded systems while establishing a compact footprint for lightweight continual learning. |
| **Continual Paradigm** | Class-Incremental Learning (CIL) with shared backbone and single expandable output head | Reflects real-world clinical deployment where task IDs are unavailable at test inference time, requiring a unified prediction across all classes seen to date. |
| **Primary Method** | Learning without Forgetting (LwF) via modified distillation loss ($M=0$) | Enables continual adaptation to novel tumor classes without storing historical patient MRI slices, respecting medical data privacy regulations (e.g., HIPAA/GDPR). |
| **Evaluated Methods**| Naive Fine-Tuning (lower bound), Experience Replay ($M=60$ and $M=120$ images), and Joint Training (upper bound oracle) on the Custom CNN | Provides a comprehensive comparison between continual learning methods: catastrophic forgetting lower bound, tiny-buffer exemplar replay, and ideal non-continual upper bound. |
| **Replay Buffer Rule** | Fixed total buffer of 60 or 120 images total, class-balanced random selection filled from each finished task; memory cost reported in metrics | Standardizes exemplar storage to directly assess the performance gain of storing 15–30 images per class versus zero-storage privacy-preserving LwF. |
| **Task Presentation Orders** | Order A: `{glioma, meningioma} -> {pituitary} -> {notumor}`; Order B: `{notumor, pituitary} -> {glioma} -> {meningioma}` | Probes continual learning robustness against reverse pathology presentation and exposes representation retention asymmetries. |
| **Reproducibility** | Fixed 2-seed evaluation (`42`, `123`) with deterministic flags enabled | Balances statistical variance estimation ($mean \pm std$) with computational efficiency across the 18-run matrix (seed `2024` marked unused). |
| **Hyperparameter Optimization** | Optional Optuna study over validation Macro-F1 | Provides optional Bayesian-guided search for distillation temperature $T$ and loss weight $\lambda$ without gating baseline runs. |
| **Deployment Format** | ONNX export with runtime validation and local Gradio GUI | Delivers platform-agnostic portable inference and an intuitive interactive visual demonstration for clinicians. |

---

## 2. Scope Change Record (Phase 0 Scope Freeze)

| Date | Scope Modification | Rationale |
|:---|:---|:---|
| **2026-10-23** | **ONE Model Only: Custom CNN Built from Scratch (Drop ResNet18 and all comparison models)** | Focuses the research strictly on comparing continual learning methods (Naive FT, LwF, ER-60, ER-120, Joint) on a dedicated lightweight architecture (<5M params, target roughly 1M to 3M). Eliminates external pretraining variables to ensure clean methodological attribution. Streamlines the experiment matrix to 18 total runs. |
| **2026-10-23** | **Drop VGG16 Architecture** | VGG16 has ~138M parameters and an outdated fully-connected architecture, making it disproportionately heavy and less relevant for modern edge-device medical continual learning. |
| **2026-10-23** | **Reduce Seeds from 3 to 2 (`[42, 123]`)** | Preserves statistical variance reporting ($mean \pm std$) while reducing total computational load. Seed `2024` split files, if generated, remain in place but are marked unused. |
| **2026-10-23** | **Remove Hybrid LwF+Replay Extension** | Keeps scientific focus sharply on comparing pure distillation ($M=0$) against discrete exemplar replay buffers ($M=60, 120$), naive fine-tuning ($M=0$), and joint training ($M=0$). |
| **2026-10-23** | **Fix Replay Buffers at 60 and 120 Images** | Establishes realistic tiny memory budgets (15–30 images per class at completion) with class-balanced random selection, tracking memory cost ($M$) as an explicit metric next to accuracy and forgetting. |
| **2026-10-23** | **Finalize 18-Run Experiment Matrix** | 4 incremental methods (`finetune`, `lwf`, `replay60`, `replay120`) $\times$ 2 orders (`Order A`, `Order B`) $\times$ 2 seeds (`42`, `123`) = 16 runs, plus Joint $\times$ 2 seeds = 2 runs $\implies$ **18 total runs**. Optuna tuning made optional. |

---

## 3. Standard Run Folder and Artifact Convention

Every experimental execution writes to an isolated folder under `results/runs/` adhering to the mandatory naming convention:

```
results/runs/<method>_<order>_<seed>_<timestamp>/
```

### Folder Contents
1. `config_snapshot.yaml`: Complete, frozen YAML configuration used for the run.
2. `train_log.csv`: Epoch-by-epoch loss (task loss, distillation loss, total loss) and validation metrics.
3. `events.out.tfevents.*`: Native TensorBoard event log for live scalar and distribution tracking.
4. `metrics.json`: Per-task test evaluations ($R_{i,j}$ matrix), final macro metrics, continual learning metrics ($A_{\text{avg}}, F, \text{BWT}$), and replay memory cost ($M$).
5. `checkpoints/`: Model weights:
   - `best.pt`: Best validation checkpoint.
   - `last.pt`: Final epoch checkpoint.
   - `task_1.pt`, `task_2.pt`, `task_3.pt`: Frozen snapshots captured at the conclusion of each incremental task.
6. `predictions/`: Serialized test logits, softmax probabilities, and ground truth arrays for post-hoc confusion matrix and ROC generation.
7. `env_info.txt`: Package versions, OS, CPU, CUDA device, and Git commit hash at invocation time.

---

## 4. Known Project Risks & Mitigation Protocols

### Risk 1: Inherent Weakness of Pure LwF in Class-Incremental Setups
- **Description**: Standard LwF was originally introduced for multi-task / domain-incremental learning with task-specific heads. In class-incremental learning with a unified output head, the lack of old-class exemplars often induces an "inter-task class bias", where the model skews predictions heavily toward recently observed classes.
- **Mitigation Strategy**:
  1. We evaluate temperature scaling $T \in [1.0, 6.0]$ to soften teacher distributions.
  2. We directly benchmark LwF ($M=0$) against tiny-buffer Experience Replay ($M=60$ and $M=120$ images) to precisely quantify the incremental benefit of exemplar storage vs. pure distillation on the custom CNN.
  3. We apply classifier head weight normalization (cosine normalization or weight alignment) if severe logit bias is observed.

### Risk 2: Near-Duplicate Data Leakage Across Merged Subsets
- **Description**: The Nickparvar dataset is compiled from three external sources (Figshare, SARTAJ, and Br35H). In medical MRI datasets, volumetric slice series from the same patient can be split across train and test partitions, resulting in near-duplicate leakage and artificially inflated test scores.
- **Mitigation Strategy**:
  1. In Phase 1 Data Audit, run perceptual hashing (`pHash`) and pairwise structural similarity (`SSIM`) scans across the official train and test sets.
  2. Flag and record any image pairs with hamming distance $\le 5$ or $\text{SSIM} \ge 0.95$.
  3. Maintain an audit report in `data/interim/leakage_audit.json` to quarantine or document duplicate cases.

### Risk 3: Model Parameter Budget Constraint (<5,000,000 parameters)
- **Description**: Adding dense fully connected layers or excessive convolutional feature maps can inadvertently breach the strict 5M parameter cap.
- **Mitigation Strategy**:
  1. Employ Global Average Pooling (GAP) after Conv Block 4, reducing feature maps to a 128-dimensional vector before classification, keeping the target between 1M and 3M parameters.
  2. Incorporate automated unit tests (`tests/test_models.py`) that strictly assert `sum(p.numel() for p in model.parameters()) < 5_000_000` on every build.

### Risk 4: Class Imbalance Distortions
- **Description**: The dataset contains uneven representation (`notumor` has ~2,000 images, whereas `glioma` has ~1,621). Incremental introduction of `notumor` with larger sample counts may overwhelm previous tumor classes.
- **Mitigation Strategy**:
  1. Implement class-weighted cross-entropy loss or balanced batch sampling during task training.
  2. Evaluate models using Macro-F1 and balanced accuracy alongside raw top-1 accuracy.

---

## 5. Phase 1 Data Engineering Decisions & Audited Ground Truth

| Parameter / Aspect | Frozen Specification | Justification & Implementation Details |
|:---|:---|:---|
| **Raw Dataset Ground Truth** | 7,200 total valid images (5,600 Training, 1,600 Testing). Exactly 1,400 train and 400 test per class. | Discovered during Task 1 disk scan. Perfectly balanced classes across both splits. Raw files are strictly immutable. |
| **Global Label Mapping** | `glioma: 0`, `meningioma: 1`, `notumor: 2`, `pituitary: 3` | Fixed permanent IDs across all task heads, datasets, and metrics. |
| **Input Resolution & Channels** | $150 \times 150$, 3 channels (grayscale replicated to RGB) | Matches proposal specification and enables standard 3-channel convolutional feature extractors. |
| **Contour Cropping & Fallback** | Gaussian blur ($k=5$), Otsu threshold, largest contour with 2% margin. Fallback to center crop if area < 20% or touches all 4 edges. | Isolates intracranial brain tissue from black background margins. Measured fallback rate across all 7,200 images is **2.06%** (148 images). |
| **Normalization Choice** | ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Standardized benchmark scaling; yields zero-centered, unit-variance distributions across replicated channels, improving gradient stability. |
| **Duplicate Isolation in Splits** | Disjoint-set clustering using SHA-256 and 64-bit DCT pHash (Hamming distance $\le 4$) | Guarantees that near-duplicate or identical patient slices within training are kept intact within either train or val, preventing split leakage. |
| **Train / Val Partition** | Stratified 15% validation: 840 val images (210/class), 4,760 train images (1,190/class) per seed (`42`, `123`). | Provides unbiased per-task hyperparameter evaluation and early stopping. Test split (1,600 images) remains strictly held out. |
| **Hardware Compatibility** | Upgraded to PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128 | Required for native kernel support on the RTX 5060 Laptop GPU (Blackwell `sm_120`), eliminating PyTorch 2.6.0 cu124 missing kernel errors. |


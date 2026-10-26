# Decisions and Risks Log (`docs/DECISIONS.md`)

This log records all architectural, algorithmic, and engineering decisions frozen for this project, along with their concise one-line justifications, scope change records, and identified risk mitigations.

---

## 1. Frozen Architectural and Engineering Decisions

| Category | Decision | One-Line Justification |
|:---|:---|:---|
| **Framework** | Standard native PyTorch (without PyTorch Lightning) | Pure PyTorch provides explicit, transparent control over multi-head dynamic expansion, teacher model snapshotting, and low-level distillation loss calculations required by LwF without framework abstraction overhead. |
| **Dataset Source** | Masoud Nickparvar Brain Tumor MRI Dataset (7,023 images) | Recognized standardized benchmark combining 4 primary classes (`glioma`, `meningioma`, `pituitary`, `notumor`) with balanced distribution and established official partitions. |
| **Split Strategy** | Official test set (1,311 images) untouched; stratified 15% validation split (~857 images) carved from official train set (5,712 images) | Prevents test set contamination and data snooping while ensuring unbiased validation across all 4 classes. |
| **Model Selection** | **ONE model only**: Custom CNN built from scratch (<5M parameters, target roughly 1M to 3M; 4 conv blocks with BatchNorm, ReLU, MaxPool; Global Average Pooling and an expandable head) | Focuses investigation strictly on how continual learning methods perform on a dedicated lightweight architecture designed from scratch for resource-constrained MRI classification. |
| **Parameter Budget** | Hard cap strictly below 5,000,000 parameters (target roughly 1M to 3M) | Ensures execution in resource-constrained environments while establishing a compact footprint for lightweight continual learning. |
| **Continual Paradigm** | Class-Incremental Learning (CIL) with shared backbone and single expandable output head | Reflects practical deployment where task IDs are unavailable at test inference time, requiring a unified prediction across all classes seen to date. |
| **Primary Method** | Learning without Forgetting (LwF) via modified distillation loss ($M=0$) | Enables continual adaptation to novel tumor classes without storing historical patient MRI slices, respecting medical data privacy regulations (e.g., HIPAA/GDPR). |
| **Evaluated Methods**| Naive Fine-Tuning (lower bound), Experience Replay ($M=60$ and $M=120$ images), and Joint Training (upper bound oracle) on the Custom CNN | Provides a comprehensive comparison between continual learning methods: catastrophic forgetting lower bound, tiny-buffer exemplar replay, and ideal non-continual upper bound. |
| **Replay Buffer Rule** | Fixed total buffer of 60 or 120 images total, class-balanced random selection filled from each finished task; memory cost reported in metrics | Standardizes exemplar storage to directly assess the performance gain of storing 15–30 images per class versus zero-storage privacy-preserving LwF. |
| **Task Presentation Orders** | Order A: `{glioma, meningioma} -> {pituitary} -> {notumor}`; Order B: `{notumor, pituitary} -> {glioma} -> {meningioma}` | Probes continual learning robustness against reverse pathology presentation and exposes representation retention asymmetries. |
| **Reproducibility** | Fixed 2-seed evaluation (`42`, `123`) with deterministic flags enabled | Balances statistical variance estimation ($mean \pm std$) with computational efficiency across the 18-run matrix (seed `2024` marked unused). |
| **Hyperparameter Optimization** | Optional Optuna study over validation Macro-F1 | Provides optional Bayesian-guided search for distillation temperature $T$ and loss weight $\lambda$ without gating baseline runs. |
| **Deployment Format** | ONNX export with runtime validation and local Gradio GUI | Delivers portable inference and an interactive visual demonstration for experimentation. |

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

---

## 6. Phase 2 Custom CNN Architectural Decisions & Known Limitations

| Parameter / Aspect | Frozen Specification | Justification & Implementation Details |
|:---|:---|:---|
| **Dataset Selection & Ground Truth** | 7,200-image balanced dataset (5,600 train / 1,600 test; 1,400/400 per class) | Deliberate decision: provides a perfectly balanced class distribution ($1:1:1:1$) across both splits, avoiding training bias from artificial skew in the older 7,023-image distribution. |
| **Known Scientific Limitation: Near-Duplicate Slices** | 2,022 cross-split near-duplicate pairs ($\text{pHash Hamming} \le 4$) exist between Train and Test | Patient slice continuity across the merged source collections means official test set evaluation may yield optimistic absolute accuracy scores. Research focus is strictly on **relative comparison** between continual learning methods (FT vs. LwF vs. Replay vs. Joint) on identical splits. |
| **Architecture Family** | 4 conv blocks (`Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2`) $\to$ GAP $\to$ `Dropout(0.4)` $\to$ `ExpandableLinearHead` | Lightweight, compact topology without dense flattening layers, maximizing spatial parameter efficiency. |
| **Width Variants Suite** | `small` (241.7k params), `medium` (1.04M params), `large` (2.53M params) | Provides three controlled capacity levels within the target range (<5M cap). Main variant selection is deferred to Phase 3 based on validation results. |
| **Expandable Classification Head** | Linear head with `expand(n_new_classes)` copying previous weights bit-identically | Prevents artificial parameter perturbation of previously learned classes when introducing novel tumor categories. Verified via `torch.equal(old_w, new_w[:old_c])`. |
| **Distillation Interface** | `forward_old_logits()` and `get_old_logits()` exposed on model | Provides clean slicing of old-class logits for LwF teacher distillation without coupling architecture code to training loops. |

---

## 7. Phase 3 Training Pipeline & Variant Selection Decisions

| Parameter / Aspect | Frozen Specification | Justification & Implementation Details |
|:---|:---|:---|
| **Main-Variant Selection Rule** | Smallest variant whose mean validation Macro-F1 across seeds `[42, 123]` is within 1 percentage point ($\le 0.01$) of the top-performing variant | Follows Occam's razor: penalizes unnecessary architectural bloat and preserves edge efficiency when larger models yield marginal predictive gains. Evaluated strictly on validation data. |
| **Selected Main Architecture** | **`medium`** variant selected (`active_variant: "medium"`, 1,036,548 params, 1077.56 MFLOPs) | `medium` achieved **0.9313 ± 0.0093** val Macro-F1 across seeds 42 and 123 under the 30-epoch fixed cosine schedule, placing within 0.75 percentage points of `large` (0.9388 ± 0.0101) and outperforming `small` (0.9189 ± 0.0100). By Occam's razor, `medium` is the smallest candidate within 1 pp of the top score, providing 4x faster training than `large` while preserving compact parameter efficiency for continual learning. |
| **Stage A Convergence Schedule** | Fixed 30 epochs with cosine decay to $10^{-6}$ over 30 epochs, no early stopping, saving both `best.pt` and `last.pt`. | Resolves high-LR oscillation and premature early stopping observed in Phase 3. Both best and last checkpoints are evaluated on val and test sets. |
| **Diagnostic: 'aug' vs. Non-'aug' Test Slices** | Seed 42: aug error rate = 6.80% vs non-aug = 12.29%. Seed 123: aug error rate = 7.77% vs non-aug = 11.02%. Per-class recall: glioma ~0.70, meningioma ~0.87, notumor ~0.998, pituitary ~0.976. | Quantifies near-duplicate slice leakage identified in Phase 1 audit. Slices with 'aug' in their filenames exhibit ~4-5 percentage point lower error rates due to training split augmentation similarity. |
| **Per-Task Epoch Budget E** | Fixed **`E = 30`** epochs for all incremental tasks across all methods | Calculated from 3-epoch moving average of validation Macro-F1 (reaching within 0.5 pp of max at epochs 30 and 29 for seeds 42 and 123, rounded up to nearest multiple of 5). |
| **BatchNorm Continual Policy** | Standard BatchNorm (momentum=0.1, running stats update freely across all tasks). Config flag: `bn_policy: "standard"`. | Preserves native domain normalization while tracking inter-task feature shifts. Flag `frozen_after_task1` is reserved for future ablation exploration. |
| **Hardware Correction** | Workstation CPU is Intel Core i7-14650HX (16 cores, 24 threads) | Corrected from initial placeholder text across documentation and profiling logs. |
| **DataLoader Multiprocessing on Windows** | Configured `num_workers: 4` in YAML, executed with `num_workers: 0` for Windows scripts | Empirical benchmark on Windows 11 showed `num_workers=0` completed epochs in 8.8s vs. 20.0s for `num_workers=4`, eliminating process spawn and pipe serialization latency. |
| **Pluggable Loss Interface** | `execute_loss_callable` accommodates `(logits, targets)`, `(model, logits, images, targets)`, or `(model, images, targets)` | Allows Phase 4 LwF knowledge distillation loss to be plugged directly into `fit()` without modifying the core training loop. |
| **Mixed Precision (AMP)** | Native `bfloat16` autocast on RTX 5060 Laptop GPU (`sm_120`) | Provides ~2x memory compression and accelerated tensor core throughput without gradient underflow issues associated with float16. |

---

## 8. Phase 4 Class-Incremental Learning Decisions

| Parameter / Aspect | Frozen Specification | Justification & Implementation Details |
|:---|:---|:---|
| **Base Architecture** | Active variant `medium` Custom CNN (1.04M params) | Selected from Stage A joint convergence fix; under 5M limit with dynamic linear head. |
| **Inference Protocol** | Class-incremental setting: argmax over all classes seen so far, no task id supplied at test time. | Standard class-incremental formulation where incoming MRI scans possess unknown pathological category. |
| **Training Budget per Task** | Exactly $E = 30$ fixed epochs, cosine decay to $10^{-6}$, keep LAST epoch's weights (`restore_best=False`), NO early stopping. | Eliminates reliance on old-class data for early stopping or model selection; standard continual learning rigor. |
| **Validation Monitoring** | Current task's new classes logged for tracking only; never used for checkpoint selection or tuning. | Strict zero-leakage protocol preventing retroactive optimization against unseen old-class retention. |
| **Learning Rate Schedule** | Task 1: $lr = 10^{-3}$; Tasks 2 & 3: $lr = 3 \times 10^{-4}$. Fresh AdamW + cosine schedule with warmup (2 epochs) per task. | Higher LR learns foundational features in Task 1; moderated LR prevents catastrophic representational collapse in Tasks 2 & 3. |
| **Head Expansion** | `ExpandableLinearHead.expand(n_new)` appends new outputs before each task, copying previous weights bit-identically. | Ensures zero artificial parameter perturbation before the new task begins. |
| **Naive Fine-Tuning** | Cross-entropy over all seen class logits; no old data, no distillation ($M=0$). | Establishes the lower bound baseline for catastrophic forgetting. |
| **Learning without Forgetting (LwF)** | Deep-copy model before task $t>1$ as frozen teacher (eval mode, no gradients). Composite loss: $\text{Loss} = \text{CE} + \lambda \cdot KD$ where $KD = T^2 \cdot \text{KL}(\text{Softmax}(\text{teacher}/T) \parallel \text{Softmax}(\text{student}/T))$. $T=2.0, \lambda=1.0$ constant. $M=0$. | Replays teacher soft probabilities on augmented new-task batches to constrain representation drift without exemplar storage. |
| **Experience Replay ($M=60, 120$)** | Fixed total capacity $M$ split evenly across seen classes ($M/k$ per class). Populated from TRAIN split only. Union batch of 64 new + $\min(32, M)$ buffer exemplars. | Privacy-conscious benchmark evaluating exemplar retention against HIPAA/GDPR storage constraints. |
| **Joint Upper Bound** | Reuses Stage A Joint Training runs of `medium` variant (Seeds 42 and 123). $M=0, F=0.0$. | Established performance ceiling on 4 classes simultaneously. |
| **Continual Sanity Check Verification** | All 4 sanity rules passed: 1. Naive Fine-Tuning exhibits catastrophic forgetting ($F = 0.9688 \gg 0.30$); 2. Experience Replay drastically outperforms Fine-Tuning ($F_{\text{ER120}} = 0.3696$ vs $F_{\text{FT}} = 0.9688$, $A_{\text{avg}} = 0.8135$ vs $0.5902$); 3. Larger memory buffer ($M=120$) consistently cuts forgetting compared to $M=60$ ($0.3696$ vs $0.5336$); 4. Joint Training provides upper bound ($A_{\text{test}} = 0.8862$, Macro-F1 = $0.8831$). | Empirical validation confirming correct implementation of continual learning strategies and absence of information leakage across tasks. |
---

## 9. Stage D: Improving LwF Within the No-Stored-Images Constraint

### 9.1 Context and Scientific Hypotheses
Original LwF equals Naive Fine-Tuning in Orders A and B (final accuracy 25%, forgetting 0.94-1.00).
- **Hypothesis 1 (Logit Shift Invariance)**: Standard softmax KD over the old logits is invariant to a uniform additive scalar shift of all old logits:
  $$\text{Softmax}\left(\frac{z_{\text{old}} + c}{T}\right) = \text{Softmax}\left(\frac{z_{\text{old}}}{T}\right)$$
  Consequently, standard cross-entropy on new-task batches can push all old logits arbitrarily downward ($z_{\text{old}} \to -\infty$) at zero distillation cost ($\mathcal{L}_{\text{KD}} = 0$), destroying the calibration between old and new class logits and creating extreme task-recency bias.
- **Hypothesis 2 (Single-Class Task Collapse)**: In Orders A and B, Tasks 2 and 3 introduce single classes (e.g. pituitary in T2, notumor in T3). In single-class adaptation without memory, cross-entropy loss trivially degenerates into predicting the new class exclusively.
- **Order C Formulation**: A balanced 2-task sequence ($T_1 = \{\text{glioma}, \text{meningioma}\} \to T_2 = \{\text{pituitary}, \text{notumor}\}$) where Task 2 introduces two classes simultaneously, testing whether single-class vs. multi-class incremental steps alter distillation dynamics.

### 9.2 Candidate LwF Architectural Variants (Zero Stored Images)
1. **`lwf_all` (Baseline)**: CE over all seen logits + $\lambda \cdot \text{softmax-KD} (T=2.0)$ on old logits.
2. **`lwf_new`**: CE over new-class logits only + $\lambda \cdot \text{softmax-KD} (T=2.0)$ on old logits.
3. **`lwfL` (Variant V2)**: CE over all seen logits + $\lambda \cdot \text{MSE}(\text{student\_old\_logits}, \text{teacher\_old\_logits})$ on the new-task batch (teacher in eval mode, no gradients). Penalizes absolute logit depression.
4. **`lwfLF` (Variant V3)**: Variant V2 + $\mu \cdot (1 - \text{cosine similarity})$ between student and teacher GAP pooled features on the same batch ($\mu=1.0$ fixed).
5. **`lwfLF_frz` (Variant V4)**: Variant V3 + freeze conv blocks 1 and 2 after Task 1 (parameters `requires_grad=False` AND BatchNorm layers kept in eval mode so running stats do not drift). *Deliberate policy deviation*: this deviates from the standard-BN policy of Phase 3 to preserve early spatial edge/texture representations.

### 9.3 Frozen Screening Selection Rule (Registered Prior to Execution)
- **Selection Metric**: The configuration with the highest mean final Macro-F1 on all seen classes across the two screening runs (Order A and Order C) measured on the **VALIDATION split of seed 42** (which includes old-class validation images).
- **Design-Time Limitation**: Old-class validation labels are used exclusively for this design-time hyperparameter choice and **never during training or per-task early stopping**. This is recognized as an explicit methodological limitation.
- **Tie-Breaking Rule**: Ties go to the simpler architectural variant ($\text{V2} < \text{V3} < \text{V4}$), then to the smaller regularization strength ($\lambda=1 < \lambda=10$).
- **Test Set Isolation**: The test set is evaluated and logged for all 12 screening runs for complete transparency, but is **strictly forbidden from being used for selection**.

### 9.4 Shared-Task-1 Protocol on Order C
For Order C ($T_1 = \{\text{glioma}, \text{meningioma}\}$), Task 1 is trained **ONCE per seed** (CE, 30 epochs, lr $10^{-3}$) and saved to disk. Every method evaluated on Order C (`finetune`, `lwf_all`, `lwf_new`, `replay60`, `replay120`, and `lwfPlus`) starts Task 2 from that exact bit-identical checkpoint.

---

## 10. Stage E and Phase 5: Metric Clean-Up, Order D, Final Analysis & Explainability

### 10.1 Diagnostic Metric Verification and Placeholder Resolution (Stage E1)
- **Initial Table Discrepancy**: In earlier Stage D summaries, Order A and B rows for `finetune` and baseline `lwf` reported `Restricted Old Acc` and `Mean Old Mass` as exactly `0.0000`, while the `Joint` row displayed `1.0000`.
- **Root-Cause Verification**:
  1. The restricted old-class accuracy diagnostic was designed in Stage D, whereas Phase 4 runs were executed prior to its introduction. The table generator defaulted missing dictionary keys via `.get("restricted_old_accuracy", 0.0)`, falsely printing `0.0000`.
  2. The `Joint` row contained hardcoded `1.0000` values, which was an artificial placeholder since Joint Training learns all 4 classes simultaneously and has no concept of "old" vs. "new" classes.
- **Resolution**:
  1. Physical checkpoints (`checkpoint_task3.pt`) were confirmed to exist on disk for all 16 Phase 4 runs.
  2. A retroactive evaluation script was executed across all saved checkpoints on the test split, computing true measured values and updating all `metrics.json` files:
     - `finetune` (Orders A & B): `restricted_old_acc = 0.3333 - 0.3342`, `mean_softmax_mass_old = 0.0004 - 0.0005`, `new_mass = 0.9996`.
     - `lwf` (Orders A & B): `restricted_old_acc = 0.3333`, `mean_softmax_mass_old = 0.0010 - 0.0013`, `new_mass = 0.9987 - 0.9990`.
     - `replay60`: Order A `restricted_old_acc = 0.6234`, `old_mass = 0.5507`; Order B `restricted_old_acc = 0.8546`, `old_mass = 0.3563`.
     - `replay120`: Order A `restricted_old_acc = 0.6942`, `old_mass = 0.6871`; Order B `restricted_old_acc = 0.8821`, `old_mass = 0.5046`.
  3. The table generator (`src/evaluation/tables/generate_tables.py`) was permanently refactored: Joint explicitly reports `n/a`, and any run missing diagnostic metrics formats strictly as an em-dash (`—`), never defaulting to `0.0000`.

### 10.2 Order D Formulation (Stage E2)
- **Task Order D**: Control-first 2-task split:
  - Task 1 ($T_1$): `notumor` (2) and `pituitary` (3) [2 classes, head size 2].
  - Task 2 ($T_2$): `glioma` (0) and `meningioma` (1) [2 new classes, head size 4].
- **Experimental Protocol**:
  - Matches Order C: Task 1 trained ONCE per seed (`shared_task1_order_d_seed{seed}.pt`, 30 epochs, lr $10^{-3}$) and shared by all methods.
  - Task 2 trained for 30 epochs with lr $3 \times 10^{-4}$, standard BN, last-epoch weights, no early stopping.
  - Zero old-class validation or test data accessed during training or model selection.
  - Methods evaluated: `finetune`, `lwf_all`, `lwf_new`, `replay60`, `replay120` across seeds `[42, 123]` (10 runs). `lwfPlus` excluded following negative screening ablation.

### 10.3 Scientific Disclosures & Methodological Honesty (Stage E3)
1. **Negative Ablation of `lwfPlus` (Variants V2–V4)**:
   - In Stage D screening, validation Macro-F1 across variants V2–V4 clustered between 0.206 and 0.221, indistinguishable from baseline `lwf_all` (0.221).
   - On the held-out test set, `lwfPlus` achieved no improvement over `lwf_all` in Orders A, B, or C.
   - We explicitly record this as a negative ablation: architectural feature distillation, lower-block freezing, and loss balancing tuned on validation differences of ~0.0003 had zero test impact.
2. **Post-Hoc Nature of `lwf_new`**:
   - `lwf_new` was devised post-hoc after inspecting the catastrophic logit collapse in Orders A and B and observing that old-class representations remained intact in Order C under the restricted accuracy diagnostic.
   - Crucially, `lwf_new` is mathematically and practically inapplicable to single-class tasks (Orders A and B) because cross-entropy over a single class is identically zero ($\log 1 = 0$).
3. **Statistical Significance Bounds (2-Seed Limit)**:
   - All experiments use $N = 2$ seeds (`[42, 123]`).
   - We strictly adhere to the $2\sigma$ rule: no performance difference is claimed as a consistent advantage unless $|\Delta\mu| > 2\sqrt{\sigma_1^2 + \sigma_2^2}$. Differences within the noise floor are treated as comparable.
4. **Fair Side-by-Side Comparison of `lwf_new` vs Replay**:
   - We avoid claiming that `lwf_new` "beats" Experience Replay unless supported across all 4 metrics ($A_{\text{avg}}$, Final Acc, Final Macro-F1, Forgetting $F$).
   - In Order C, `replay120` achieves higher Final Accuracy ($0.6993$ vs $0.6022$) and higher $A_{\text{avg}}$ ($0.7834$ vs $0.7349$), while `lwf_new` matches its forgetting resistance ($0.4681$ vs $0.4688$) with zero stored images.

### 10.4 Explainability via Grad-CAM (Phase 5 P2)
- **Target Conv Layer**: The 4th convolutional block (`model.backbone.blocks[3]`).
- **Models Evaluated on Order C Seed 42**:
  1. Task 1 Baseline (`shared_task1_order_c_seed42.pt`)
  2. Naive Fine-Tuning (`checkpoint_task2.pt`)
  3. LwF Standard / All (`checkpoint_task2.pt`)
  4. LwF New-Class CE (`checkpoint_task2.pt`)
  5. Experience Replay $M=120$ (`checkpoint_task2.pt`)
  6. Joint Training Upper Bound (`best.pt`)
- **Evaluation Set**: 16 deterministic unaugmented test images (4 per diagnostic class).
- **Outputs**: Master comparison grid (`results/figures/gradcam/gradcam_comparison_grid.png`) and class-specific panels (`results/figures/gradcam/gradcam_class_*`).
- **Interpretation & Limitations**: Grad-CAM overlays were generated for 16 test samples (4 per class) across methods. The visualizations produce spatial attention heatmaps across the test images. A qualitative comparison reveals differences in activation localization across methods; however, Grad-CAM heatmaps require visual inspection by a human and cannot rule out shortcut learning or artifact reliance across the dataset.






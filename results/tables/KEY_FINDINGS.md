# Comprehensive Key Findings & Scientific Analysis: Class-Incremental MRI Classification

This report documents the empirical and theoretical findings of class-incremental learning for Brain Tumor MRI classification using Custom CNN under Learning without Forgetting (LwF), Experience Replay, and Naive Fine-Tuning across multiple task orderings and experimental configurations.

---

## 1. Summary of Major Empirical Findings

### Finding 1: Catastrophic Forgetting in Standard Continual Learning
In standard task formulations with single-class incremental additions (Orders A and B: head expanding $2 \to 3 \to 4$), **Naive Fine-Tuning suffers complete catastrophic forgetting**, collapsing to 25.00% final accuracy (random guess) and a Macro-F1 of 0.1000 (only predicting the latest seen class).
Standard **Learning without Forgetting (LwF)** fails to mitigate this collapse in Orders A and B, matching Naive Fine-Tuning identically (25.00% accuracy, 0.1000 Macro-F1, forgetting $F = 0.94 - 1.00$).

### Finding 2: Root-Cause Diagnosis of LwF Failure via Restricted-Class Analysis
Through our retroactive diagnostic evaluation of saved model checkpoints:
- **Restricted Old-Class Accuracy**: When evaluating test images from old classes while restricting the prediction argmax strictly to the old-class logits (masking new classes), the model achieves:
  - Orders A & B: ~33.33% (random guess among the 3 old classes, indicating representations for earlier classes have eroded).
  - Order C: **85.63%** restricted accuracy! The feature representations and linear decision boundary between the old classes (`glioma` vs. `meningioma`) were thoroughly preserved.
- **Old-vs-New Softmax Mass**: However, the mean softmax probability mass allocated to the old classes on those same images was only **0.0040 (0.4%)**!
- **Theoretical Mechanism**: Standard LwF applies cross-entropy loss over all seen classes on new-task images. Because the softmax function $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ and knowledge distillation loss $\mathcal{L}_{\text{KD}}$ are invariant to a uniform shift of all old logits ($z_{\text{old}} \to z_{\text{old}} - c$), the optimizer can minimize cross-entropy on new data by uniformly suppressing all old logits without incurring any distillation penalty. This drives the softmax probability of all past classes to near zero at inference time.

### Finding 3: The Role and Limits of `lwf_new` (New-Class Cross-Entropy)
To prevent logit suppression, `lwf_new` computes cross-entropy strictly over the new-class logit slice ($z_{\text{new}}$) while preserving old logits solely via knowledge distillation:
- In Order C, `lwf_new` restores balanced softmax mass allocation to **0.4718 (47.2%)**, preventing the old classes from being drowned out by the new classes.
- This raises final test Macro-F1 from **0.3389** (LwF All) to **0.5904** and final accuracy from **50.00%** to **60.22%**, cutting forgetting from **0.8675** to **0.4681**.
- **Post-hoc Limitation**: `lwf_new` was designed post-hoc based on the diagnostic findings in Order C. It is mathematically and practically **inapplicable to single-class tasks** (Orders A and B) because cross-entropy over a single class is identically zero ($\log 1 = 0$).

---

## 2. Rigorous Scientific Disclosures & Negative Ablations

### Disclosure A: Negative Ablation of `lwfPlus` Variants (V2–V4)
In Stage D, architectural modifications were screened to improve distillation:
- **Variant V2 (`lwfLF_l1`)**: Less-forgetting loss balancing with weight $\lambda = 1.0$.
- **Variant V3 (`lwfLF_l10`)**: Loss balancing with $\lambda = 10.0$.
- **Variant V4 (`lwfLF_frz_l1`)**: Freezing Conv Blocks 1 & 2 with BatchNorm in eval mode + Less-forgetting loss.
- **Empirical Result**: During validation screening, Macro-F1 across all variants clustered tightly between **0.206 and 0.221**, essentially indistinguishable from baseline LwF (0.221). On the held-out test set, `lwfPlus` achieved:
  - Order A: Acc 0.2500, Macro-F1 0.1000, Forgetting 0.9338 (vs LwF 0.9356).
  - Order B: Acc 0.2500, Macro-F1 0.1000, Forgetting 1.0000.
  - Order C: Acc 0.4969, Macro-F1 0.3405, Forgetting 0.8675 (vs LwF All 0.8675).
- **Scientific Takeaway**: Hyperparameter and architectural tuning based on validation differences of ~0.0003 had zero test-set impact. This is documented transparently as a negative ablation.

### Disclosure B: Statistical Significance & Seed Count Limitations
- Experiments were evaluated on **$N = 2$ seeds** (`[42, 123]`).
- Consistent with rigorous scientific methodology, differences are not claimed as statistically significant wins unless the difference in sample means exceeds approximately two standard deviations ($|\Delta\mu| > 2\sigma$).
- For example, in Order A, the forgetting difference between `lwf` ($0.9356$) and `lwfplus` ($0.9338$) is $0.0018$, while seed standard deviation is $\pm 0.0025$. This difference is within the noise floor.

### Disclosure C: Fair Comparison of `lwf_new` vs. Experience Replay (Orders C & D Side-by-Side)
We do not make blanket claims that `lwf_new` "beats" Experience Replay. A side-by-side evaluation across all 4 continual learning metrics shows:

#### Order C ($T_1 = \{\text{glioma}, \text{meningioma}\} \to T_2 = \{\text{pituitary}, \text{notumor}\}$):
1. **$A_{\text{avg}}$ (Average Incremental Accuracy)**: Replay120 ($0.7834 \pm 0.0046$) > Replay60 ($0.7483 \pm 0.0074$) > `lwf_new` ($0.7349 \pm 0.0017$).
2. **Final Accuracy**: Replay120 ($0.6993 \pm 0.0106$) > Replay60 ($0.6290 \pm 0.0160$) > `lwf_new` ($0.6022 \pm 0.0047$).
3. **Final Macro-F1**: Replay120 ($0.6590 \pm 0.0157$) > `lwf_new` ($0.5904 \pm 0.0027$) > Replay60 ($0.5608 \pm 0.0228$).
4. **Overall Forgetting ($F$) ↓**: `lwf_new` ($0.4681 \pm 0.0081$) ≈ Replay120 ($0.4688 \pm 0.0226$) < Replay60 ($0.6094 \pm 0.0331$).

#### Order D ($T_1 = \{\text{notumor}, \text{pituitary}\} \to T_2 = \{\text{glioma}, \text{meningioma}\}$):
1. **$A_{\text{avg}}$ (Average Incremental Accuracy)**: Replay120 ($0.8638 \pm 0.0062$) > Replay60 ($0.8127 \pm 0.0069$) > `lwf_new` ($0.7954 \pm 0.0101$).
2. **Final Accuracy**: Replay120 ($0.7275 \pm 0.0125$) > Replay60 ($0.6257 \pm 0.0137$) > `lwf_new` ($0.5909 \pm 0.0203$).
3. **Final Macro-F1**: Replay120 ($0.7289 \pm 0.0124$) > Replay60 ($0.6038 \pm 0.0100$) > `lwf_new` ($0.5639 \pm 0.0246$).
4. **Overall Forgetting ($F$) ↓**: `lwf_new` ($0.1431 \pm 0.0093$) < Replay120 ($0.3557 \pm 0.0181$) < Replay60 ($0.5694 \pm 0.0206$).

**Conclusion**:
- In both 2-task scenarios (Orders C and D), `lwf_new` achieves outstanding retention of past classes, yielding the lowest or tied-lowest forgetting ($F = 0.4681$ in C, $F = 0.1431$ in D) while storing **zero exemplar images** (only retaining the 4.15 MB teacher model snapshot).
- However, Experience Replay buffers ($M=60, 120$) that store actual patient images achieve higher overall accuracy and higher $A_{\text{avg}}$ across all tasks.
- Therefore, `lwf_new` does *not* strictly beat Replay across all metrics; instead, it establishes an optimal Pareto trade-off between privacy (zero stored raw patient data) and catastrophic forgetting mitigation.

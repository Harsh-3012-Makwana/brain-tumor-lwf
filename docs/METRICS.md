# Evaluation Metrics Formulation (`docs/METRICS.md`)

This document provides formal mathematical definitions for all evaluation metrics used in this research project, covering classification performance, continual learning retention, and computational efficiency.

---

## 1. Classification Performance Metrics

For a multi-class classification setting with $C$ classes ($C \in \{2, 3, 4\}$ depending on the incremental task stage), let:
- $TP_c$: True Positives for class $c$
- $FP_c$: False Positives for class $c$
- $FN_c$: False Negatives for class $c$
- $TN_c$: True Negatives for class $c$

### 1.1 Overall Accuracy
$$\text{Accuracy} = \frac{\sum_{c=1}^{C} TP_c}{\sum_{c=1}^{C} (TP_c + FP_c + FN_c + TN_c)}$$

### 1.2 Class-Specific Precision, Recall, and F1-Score
For each class $c$:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$
$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
$$\text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2 TP_c}{2 TP_c + FP_c + FN_c}$$

### 1.3 Macro-Averaged Metrics
To treat all tumor classes equally regardless of class size imbalances:
$$\text{Macro Precision} = \frac{1}{C} \sum_{c=1}^{C} \text{Precision}_c$$
$$\text{Macro Recall} = \frac{1}{C} \sum_{c=1}^{C} \text{Recall}_c$$
$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c$$

---

## 2. Continual Learning Evaluation Metrics

Let $N = 3$ denote the total number of sequential tasks.  
We construct an evaluation accuracy matrix $R \in \mathbb{R}^{N \times N}$, where:
- $R_{i, j} \in [0, 1]$ represents the test classification accuracy evaluated on the test set of task $j$ after the model has completed training on task $i$ (for $1 \le j \le i \le N$).

```
        Task 1 Test   Task 2 Test   Task 3 Test
After T1:   R_1,1           -             -
After T2:   R_2,1         R_2,2           -
After T3:   R_3,1         R_3,2         R_3,3
```

### 2.1 Task-Specific Average Accuracy after Task $k$
The average test accuracy across all tasks seen up to and including task $k$:
$$A_k = \frac{1}{k} \sum_{j=1}^{k} R_{k, j}$$

### 2.2 Average Incremental Accuracy ($A_{\text{avg}}$)
The primary continual learning performance index, measuring the mean accuracy across the entire incremental learning progression:
$$A_{\text{avg}} = \frac{1}{N} \sum_{k=1}^{N} A_k = \frac{1}{N} \sum_{k=1}^{N} \left( \frac{1}{k} \sum_{j=1}^{k} R_{k, j} \right)$$

### 2.3 Task-Specific and Overall Forgetting ($F$)
Forgetting measures the drop in accuracy on a previous task $j$ after subsequent training on task $k$ ($k > j$):
$$f_j^k = \max_{l \in \{1, \dots, k-1\}} R_{l, j} - R_{k, j}, \quad \forall j < k$$

The overall Forgetting $F$ after learning all $N$ tasks is the average forgetting across all previous tasks:
$$F = \frac{1}{N - 1} \sum_{j=1}^{N - 1} f_j^N = \frac{1}{N - 1} \sum_{j=1}^{N - 1} \left( \max_{l \in \{1, \dots, N-1\}} R_{l, j} - R_{N, j} \right)$$
- $F \in [0, 1]$: Lower values indicate superior retention. $F = 0$ corresponds to perfect zero forgetting.

### 2.4 Backward Transfer ($\text{BWT}$)
Backward transfer measures the net retroactive influence of subsequent task learning on previous task test performance relative to when those tasks were first learned:
$$\text{BWT} = \frac{1}{N - 1} \sum_{j=1}^{N - 1} (R_{N, j} - R_{j, j})$$
- $\text{BWT} < 0$: Catastrophic forgetting has occurred on past tasks.
- $\text{BWT} = 0$: Past task performance is completely maintained.
- $\text{BWT} > 0$: Positive backward transfer (learning new classes improved previous class representations).

---

## 3. Computational Efficiency & Deployment Metrics

### 3.1 Parameter Count ($\#\text{Params}$)
Total number of trainable and non-trainable weights:
$$\#\text{Params} = \sum_{p \in \Theta} \text{numel}(p)$$
*Requirement: Must strictly satisfy $\#\text{Params} < 5,000,000$.*

### 3.2 Floating Point Operations (FLOPs / MACs)
Theoretical Multiply-Accumulate Operations (MACs) and Floating-Point Operations (FLOPs $\approx 2 \times \text{MACs}$) required for a single forward pass with input dimension $(1, 3, 224, 224)$, calculated using `fvcore` or `thop`.

### 3.3 Checkpoint Storage Footprint
Disk footprint in Megabytes (MB) of the serialized PyTorch model state dictionary (`.pt`) and exported ONNX model (`.onnx`).

### 3.4 Inference Latency
Wall-clock elapsed time required to classify a single image, averaged over 1,000 consecutive forward iterations after 100 warm-up iterations:
$$\bar{t} = \frac{1}{K} \sum_{k=1}^{K} t_k \quad (\text{ms/slice})$$
Reported as $\text{mean} \pm \text{std}$ on:
1. **CPU**: Intel Core i7-14650HX (batch size = 1)
2. **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (batch size = 1 and batch size = 32)

### 3.5 Replay Buffer Memory Cost ($M$)
Quantifies the patient image storage requirement imposed by memory-based replay, directly addressing medical data privacy (HIPAA/GDPR) constraints:
$$M \in \{0, 60, 120\} \quad (\text{number of stored raw patient images})$$
- **Naive Fine-Tuning, LwF, and Joint Training**: $M = 0$ (zero old patient images stored).
- **Experience Replay**: $M = 60$ images total or $M = 120$ images total stored across all completed tasks, sampled via class-balanced random selection.
- Reported directly alongside Accuracy, Forgetting ($F$), and $\text{BWT}$ in all comparison tables to evaluate the performance-privacy trade-off.

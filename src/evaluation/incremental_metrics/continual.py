"""Evaluation metrics for class-incremental continual learning.

Implements accuracy matrix R, Average Incremental Accuracy (A_avg),
Forgetting (F), Backward Transfer (BWT), and Replay Memory Cost (M),
strictly adhering to the formal definitions in docs/METRICS.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_continual_metrics(
    accuracy_matrix: List[List[Optional[float]]],
    memory_cost: int = 0,
) -> Dict[str, Any]:
    """Compute continual learning indices from an accuracy matrix R.

    Args:
        accuracy_matrix: N x N lower-triangular matrix R where R[i][j] is the
            test classification accuracy on the test set of task j after
            training task i (0-indexed, 0 <= j <= i < N).
        memory_cost: Count of stored raw exemplar images (0, 60, 120).

    Returns:
        Dict containing:
            - accuracy_matrix: The N x N accuracy matrix.
            - task_average_accuracies: List of A_k for k=1..N.
            - average_incremental_accuracy: A_avg scalar.
            - task_forgetting: List of forgetting values f_j^N for j < N-1.
            - overall_forgetting: Mean forgetting F.
            - backward_transfer: BWT scalar.
            - replay_memory_cost: M (images).
    """
    R = np.array(accuracy_matrix, dtype=float)
    N = R.shape[0]

    if N < 2:
        raise ValueError(f"Continual learning requires at least 2 tasks, got {N}")

    # 1. Task-Specific Average Accuracy A_k for each step k (1 <= k <= N)
    task_avg_accs: List[float] = []
    for k in range(1, N + 1):
        # Slice R[k-1, :k]
        seen_accs = R[k - 1, :k]
        a_k = float(np.mean(seen_accs))
        task_avg_accs.append(round(a_k, 4))

    # 2. Average Incremental Accuracy A_avg = (1 / N) * sum(A_k)
    a_avg = float(np.mean(task_avg_accs))

    # 3. Forgetting f_j^N = max_{l in 1..N-1}(R_l,j) - R_N,j for all j < N-1
    task_forgetting: List[float] = []
    for j in range(N - 1):
        # Column j from row 0 to N-2
        hist_accs = R[j : N - 1, j]
        max_past_acc = float(np.max(hist_accs))
        final_acc = float(R[N - 1, j])
        f_j = max(0.0, max_past_acc - final_acc)
        task_forgetting.append(round(f_j, 4))

    overall_f = float(np.mean(task_forgetting)) if task_forgetting else 0.0

    # 4. Backward Transfer BWT = (1 / (N - 1)) * sum(R_N,j - R_j,j)
    bwt_elements: List[float] = []
    for j in range(N - 1):
        initial_acc = float(R[j, j])
        final_acc = float(R[N - 1, j])
        bwt_elements.append(final_acc - initial_acc)

    bwt = float(np.mean(bwt_elements)) if bwt_elements else 0.0

    return {
        "accuracy_matrix": [[round(x, 4) if not np.isnan(x) else None for x in row] for row in R],
        "task_average_accuracies": task_avg_accs,
        "average_incremental_accuracy": round(a_avg, 4),
        "task_forgetting": task_forgetting,
        "overall_forgetting": round(overall_f, 4),
        "backward_transfer": round(bwt, 4),
        "replay_memory_cost": int(memory_cost),
    }


def compute_restricted_old_class_metrics(
    model: Any,
    test_loader: Any,
    seen_classes: List[int],
    num_old_classes: int,
    device: Any,
) -> Dict[str, Any]:
    """Compute restricted old-class accuracy and old vs new softmax probability mass.

    Args:
        model: Model with output head matching len(seen_classes).
        test_loader: DataLoader supplying test batches with global target IDs.
        seen_classes: Ordered list of global class IDs seen so far.
        num_old_classes: Count of old classes (0 to num_old_classes - 1 in head).
        device: Active torch device.

    Returns:
        Dict containing:
            - restricted_old_accuracy: Accuracy on old-class test images when argmax
              is strictly restricted to the old logits only.
            - mean_softmax_mass_old: Mean softmax probability assigned to old classes
              on old-class test images.
            - mean_softmax_mass_new: Mean softmax probability assigned to new classes
              on old-class test images.
            - total_old_samples: Total number of evaluated old-class test images.
    """
    import torch

    if num_old_classes <= 0 or num_old_classes >= len(seen_classes):
        return {
            "restricted_old_accuracy": 0.0,
            "mean_softmax_mass_old": 1.0,
            "mean_softmax_mass_new": 0.0,
            "total_old_samples": 0,
        }

    model.eval()
    old_global_set = set(seen_classes[:num_old_classes])
    old_classes_arr = np.array(seen_classes[:num_old_classes])

    correct_restricted = 0
    total_old = 0
    sum_mass_old = 0.0
    sum_mass_new = 0.0

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1].detach().cpu().numpy()

            mask = np.isin(targets, list(old_global_set))
            if not np.any(mask):
                continue

            old_targets = targets[mask]
            old_images = images[torch.tensor(mask, device=device)]

            logits = model(old_images)  # (B_old, num_seen)
            probs = torch.softmax(logits, dim=-1)

            # Softmax mass on old vs new classes
            mass_old = probs[:, :num_old_classes].sum(dim=-1).detach().cpu().numpy()
            mass_new = probs[:, num_old_classes:].sum(dim=-1).detach().cpu().numpy()

            sum_mass_old += float(np.sum(mass_old))
            sum_mass_new += float(np.sum(mass_new))

            # Restricted argmax on old logits only
            restricted_idx = torch.argmax(logits[:, :num_old_classes], dim=-1).detach().cpu().numpy()
            pred_globals = old_classes_arr[restricted_idx]

            correct_restricted += int(np.sum(pred_globals == old_targets))
            total_old += len(old_targets)

    acc_restricted = float(correct_restricted / total_old) if total_old > 0 else 0.0
    mean_old_mass = float(sum_mass_old / total_old) if total_old > 0 else 0.0
    mean_new_mass = float(sum_mass_new / total_old) if total_old > 0 else 0.0

    return {
        "restricted_old_accuracy": round(acc_restricted, 4),
        "mean_softmax_mass_old": round(mean_old_mass, 4),
        "mean_softmax_mass_new": round(mean_new_mass, 4),
        "total_old_samples": int(total_old),
    }

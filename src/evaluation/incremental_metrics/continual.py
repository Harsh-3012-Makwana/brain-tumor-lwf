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

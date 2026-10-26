"""Unit tests for Phase 5 and Stage E requirements.

Tests:
  1. lwf_new vs lwf_all loss on a synthetic example (verify lwf_new ignores old classes in CE).
  2. Order D in task builder: verify task sets, head sizes (2 -> 4).
  3. Table generator: verify that missing metrics format as '—', Joint as 'n/a', never 0.0000.
  4. Grad-CAM: verify heatmap output shape (150, 150) and values strictly within [0.0, 1.0].
  5. Plotting smoke test on synthetic metrics: verify all 6 plot generators execute cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tasks.builder import TASK_ORDERS, get_task_datasets
from src.evaluation.explainability.gradcam import GradCAM
from src.evaluation.plots.phase5_plots import (
    plot_accuracy_matrices_heatmaps,
    plot_accuracy_trajectories,
    plot_accuracy_vs_memory,
    plot_confusion_matrices,
    plot_final_metrics_bars,
    plot_restricted_acc_vs_old_mass,
)
from src.evaluation.tables.generate_tables import format_val_or_dash
from src.incremental.strategies.losses import LwFLoss, LwFNewLoss
from src.models.custom_cnn.factory import build_model
from src.utils.paths import SPLITS_DIR


def test_lwf_new_vs_lwf_all_loss_synthetic():
    """Verify that lwf_new Cross-Entropy operates strictly over new-class logits."""
    # Synthetic scenario:
    # 2 old classes (0, 1), 2 new classes (2, 3) -> 4 seen classes total
    seen_classes = [0, 1, 2, 3]
    num_old = 2

    # Teacher model (old classes)
    teacher = build_model(variant="small", num_classes=2)
    teacher.eval()

    # Synthetic student logits (batch=2, classes=4)
    student_logits = torch.tensor([
        [2.0, 1.0, 3.0, 0.5],
        [1.5, 0.5, 1.0, 4.0],
    ], requires_grad=True)

    # Targets: global labels in new classes (e.g. 2 and 3)
    targets = torch.tensor([2, 3])
    dummy_images = torch.zeros(2, 3, 150, 150)

    loss_fn_all = LwFLoss(
        teacher=teacher,
        seen_classes=seen_classes,
        num_old_classes=num_old,
    )
    loss_fn_new = LwFNewLoss(
        teacher=teacher,
        seen_classes=seen_classes,
        num_old_classes=num_old,
    )

    loss_all = loss_fn_all(None, student_logits, dummy_images, targets)
    loss_new = loss_fn_new(None, student_logits, dummy_images, targets)

    # Compute manual CE for lwf_all (softmax over all 4 classes)
    ce_all_expected = F.cross_entropy(student_logits, torch.tensor([2, 3]))

    # Compute manual CE for lwf_new (slice new logits [:, 2:], remapped targets [0, 1])
    new_logits_slice = student_logits[:, num_old:]
    remapped_targets = torch.tensor([0, 1])
    ce_new_expected = F.cross_entropy(new_logits_slice, remapped_targets)

    # Sliced CE must differ from full CE because the denominator only includes new logits
    assert ce_new_expected.item() != ce_all_expected.item()
    assert ce_new_expected.item() < ce_all_expected.item()

    # Verify total loss difference
    assert loss_new.item() != loss_all.item()
    assert loss_new.item() < loss_all.item()


def test_order_d_in_task_builder():
    """Verify Order D definition, task sets, and head expansion sizes."""
    assert "D" in TASK_ORDERS
    order_d = TASK_ORDERS["D"]
    assert len(order_d) == 2
    # Task 1: [notumor (2), pituitary (3)]
    assert order_d[0] == [2, 3]
    # Task 2: [glioma (0), meningioma (1)]
    assert order_d[1] == [0, 1]

    # Verify cumulative seen classes
    seen_t1 = order_d[0]
    seen_t2 = order_d[0] + order_d[1]
    assert len(seen_t1) == 2
    assert len(seen_t2) == 4

    # Build incremental tasks with builder
    tasks = get_task_datasets(order="D", seed=42, splits_dir=SPLITS_DIR)
    assert len(tasks) == 2

    # Task 1
    assert tasks[0].task_id == 1
    assert tasks[0].new_classes == [2, 3]
    assert tasks[0].seen_classes == [2, 3]
    assert tasks[0].head_size == 2

    # Task 2
    assert tasks[1].task_id == 2
    assert tasks[1].new_classes == [0, 1]
    assert set(tasks[1].seen_classes) == {0, 1, 2, 3}
    assert tasks[1].head_size == 4
    assert tasks[1].head_size == 4


def test_table_generator_missing_metric_formatting():
    """Verify that missing metrics format as '—' or 'n/a', and never as 0.0000."""
    # Missing / None
    assert format_val_or_dash(None) == "—"
    assert format_val_or_dash("") == "—"

    # Explicit n/a for Joint
    assert format_val_or_dash("n/a") == "n/a"
    assert format_val_or_dash("N/A") == "n/a"

    # Valid values
    assert format_val_or_dash(0.3333) == "0.3333"
    assert format_val_or_dash(0.0) == "0.0000"  # When value is truly 0.0
    assert format_val_or_dash("0.8569") == "0.8569"


def test_gradcam_output_shape_and_bounds():
    """Verify Grad-CAM output shape (150, 150) and values strictly within [0.0, 1.0]."""
    model = build_model(variant="small", num_classes=4)
    model.eval()

    gradcam = GradCAM(model, target_layer=model.backbone.blocks[3])

    # Synthetic image (1, 3, 150, 150)
    x = torch.randn(1, 3, 150, 150, requires_grad=True)

    heatmap, pred_class, pred_prob = gradcam.generate(x)
    gradcam.remove_hooks()

    # Shape verification
    assert heatmap.shape == (150, 150)
    assert isinstance(heatmap, np.ndarray)

    # Value bounds verification [0.0, 1.0]
    assert float(np.min(heatmap)) >= 0.0 - 1e-6
    assert float(np.max(heatmap)) <= 1.0 + 1e-6
    assert 0 <= pred_class < 4
    assert 0.0 <= pred_prob <= 1.0


def test_plotting_smoke_test_on_synthetic_metrics(tmp_path, monkeypatch):
    """Smoke test ensuring all 6 Phase 5 plotting functions execute without error."""
    # Monkeypatch FIGURES_DIR to tmp_path
    monkeypatch.setattr("src.evaluation.plots.phase5_plots.FIGURES_DIR", tmp_path)

    # Synthetic metrics payload covering Orders A, B, C, D
    synthetic_metrics: List[Dict[str, Any]] = []
    for order in ["A", "B", "C", "D"]:
        n_tasks = 3 if order in ["A", "B"] else 2
        for method in ["finetune", "lwf", "lwf_all", "lwf_new", "replay60", "replay120"]:
            if order in ["A", "B"] and method in ["lwf_all", "lwf_new"]:
                continue
            if order in ["C", "D"] and method == "lwf":
                continue

            for seed in [42, 123]:
                # Generate synthetic accuracy matrix
                R = np.zeros((n_tasks, n_tasks))
                for i in range(n_tasks):
                    for j in range(i + 1):
                        R[i, j] = 0.5 + 0.3 * np.random.rand()

                synthetic_metrics.append({
                    "method": method,
                    "order": order,
                    "seed": seed,
                    "average_incremental_accuracy": 0.75,
                    "final_accuracy": 0.65,
                    "final_macro_f1": 0.62,
                    "overall_forgetting": 0.35,
                    "backward_transfer": -0.35,
                    "restricted_old_accuracy": 0.80,
                    "mean_softmax_mass_old": 0.45,
                    "mean_softmax_mass_new": 0.55,
                    "accuracy_matrix": R.tolist(),
                    "final_confusion_matrix": np.ones((4, 4)).tolist(),
                    "per_task_summaries": [
                        {"task_idx": t + 1, "test_accuracy_seen": 0.70 + 0.05 * t}
                        for t in range(n_tasks)
                    ],
                })

    # Execute all 6 plots
    p1 = plot_accuracy_trajectories(synthetic_metrics)
    assert p1.is_file()

    p2 = plot_final_metrics_bars(synthetic_metrics)
    assert p2.is_file()

    p3 = plot_accuracy_matrices_heatmaps(synthetic_metrics, seed=42)
    assert p3.is_file()

    p4 = plot_confusion_matrices(synthetic_metrics, seed=42)
    assert p4.is_file()

    p5 = plot_accuracy_vs_memory(synthetic_metrics)
    assert p5.is_file()

    p6 = plot_restricted_acc_vs_old_mass(synthetic_metrics)
    assert p6.is_file()

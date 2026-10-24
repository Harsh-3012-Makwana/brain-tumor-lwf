"""Unit tests for class-incremental continual learning components."""

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.incremental_metrics.continual import compute_continual_metrics
from src.incremental.buffers.class_balanced_buffer import ClassBalancedBuffer
from src.incremental.buffers.replay_loader import ReplayDataLoader
from src.incremental.strategies.losses import CrossEntropySeenLoss, LwFLoss
from src.incremental.teachers.manager import create_frozen_teacher
from src.models.custom_cnn.factory import build_model


def test_accuracy_matrix_and_forgetting_formulas_match_hand_computed_example():
    """Verify compute_continual_metrics against exact manual math."""
    # Hand-crafted 3x3 matrix R:
    # Task 1: R_1,1 = 0.90
    # Task 2: R_2,1 = 0.70, R_2,2 = 0.85
    # Task 3: R_3,1 = 0.50, R_3,2 = 0.75, R_3,3 = 0.80
    R = [
        [0.90, None, None],
        [0.70, 0.85, None],
        [0.50, 0.75, 0.80],
    ]
    # Hand calculations:
    # A_1 = 0.90
    # A_2 = (0.70 + 0.85) / 2 = 0.775
    # A_3 = (0.50 + 0.75 + 0.80) / 3 = 0.683333...
    # A_avg = (0.90 + 0.775 + 0.683333...) / 3 = 2.358333 / 3 = 0.786111...
    # Forgetting:
    # f_1^3 = max(0.90, 0.70) - 0.50 = 0.90 - 0.50 = 0.40
    # f_2^3 = max(0.85) - 0.75 = 0.10
    # Overall F = (0.40 + 0.10) / 2 = 0.25
    # BWT:
    # (R_3,1 - R_1,1) = 0.50 - 0.90 = -0.40
    # (R_3,2 - R_2,2) = 0.75 - 0.85 = -0.10
    # Overall BWT = (-0.40 + -0.10) / 2 = -0.25
    metrics = compute_continual_metrics(R, memory_cost=60)

    assert pytest.approx(metrics["task_average_accuracies"][0], abs=1e-4) == 0.90
    assert pytest.approx(metrics["task_average_accuracies"][1], abs=1e-4) == 0.775
    assert pytest.approx(metrics["task_average_accuracies"][2], abs=1e-4) == 0.6833
    assert pytest.approx(metrics["average_incremental_accuracy"], abs=1e-4) == 0.7861
    assert pytest.approx(metrics["task_forgetting"][0], abs=1e-4) == 0.40
    assert pytest.approx(metrics["task_forgetting"][1], abs=1e-4) == 0.10
    assert pytest.approx(metrics["overall_forgetting"], abs=1e-4) == 0.25
    assert pytest.approx(metrics["backward_transfer"], abs=1e-4) == -0.25
    assert metrics["replay_memory_cost"] == 60


def test_head_expansion_keeps_old_logits_unchanged_before_training():
    """Verify expanding head preserves previous class output logits bit-identically."""
    model = build_model(variant="small", num_classes=2)
    model.eval()

    x = torch.randn(4, 3, 150, 150)
    with torch.no_grad():
        old_logits = model(x)

    # Expand head by 1 class (2 -> 3)
    model.expand(1)

    with torch.no_grad():
        new_logits = model(x)

    assert new_logits.shape == (4, 3)
    # The first 2 classes must match old_logits exactly
    assert torch.allclose(new_logits[:, :2], old_logits, atol=1e-6)


def test_teacher_parameters_identical_before_and_after_training_step():
    """Verify teacher model remains frozen in eval mode and is never updated by optimizer."""
    # Student starts with 2 classes (Task 1)
    student = build_model(variant="small", num_classes=2)
    teacher = create_frozen_teacher(student)

    # Student expands to 3 classes (Task 2)
    student.expand(1)

    # Verify teacher is in eval mode and has 2 classes
    assert not teacher.training
    assert getattr(teacher, "num_classes", 0) == 2

    # Store initial teacher weights
    initial_teacher_params = [p.clone() for p in teacher.parameters()]

    # Train student for 1 step
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    loss_fn = LwFLoss(
        teacher=teacher,
        seen_classes=[0, 1, 2],
        num_old_classes=2,
        temperature=2.0,
        lambda_kd=1.0,
    )

    x = torch.randn(8, 3, 150, 150)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    student.train()
    optimizer.zero_grad()
    logits = student(x)
    loss = loss_fn(student, logits, x, y)
    loss.backward()
    optimizer.step()

    # Verify teacher weights are strictly identical
    for p_init, p_curr in zip(initial_teacher_params, teacher.parameters()):
        assert torch.equal(p_init, p_curr), "Teacher parameter was modified during training!"
        assert p_curr.grad is None, "Teacher parameter received gradients!"


def test_replay_buffer_size_balance_and_train_split_only():
    """Verify buffer holds at most M images, is class-balanced, and uses train data only."""
    # Synthetic dataset mimicking MRIDataset item (tensor, label, path)
    class MockDataset:
        def __init__(self, class_id: int, count: int):
            self.class_id = class_id
            self.items = [
                (torch.randn(3, 150, 150), class_id, f"train_img_cls{class_id}_{i}.jpg")
                for i in range(count)
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    capacity = 60
    buffer = ClassBalancedBuffer(capacity=capacity, seed=42)

    # Task 1: 2 classes (0 and 1)
    ds_task1 = MockDataset(0, 100)
    # Combine classes 0 and 1
    ds_task1.items.extend([(torch.randn(3, 150, 150), 1, f"train_img_cls1_{i}.jpg") for i in range(100)])

    buffer.update_buffer(seen_classes=[0, 1], task_train_dataset=ds_task1)

    assert buffer.total_samples == 60
    assert len(buffer.storage[0]) == 30
    assert len(buffer.storage[1]) == 30
    # Verify all items come from train paths
    for cls_id in [0, 1]:
        for item in buffer.storage[cls_id]:
            assert "train_img" in item[2]

    # Task 2: adds class 2 -> rebalance across 3 classes (target = 60 // 3 = 20 each)
    ds_task2 = MockDataset(2, 50)
    buffer.update_buffer(seen_classes=[0, 1, 2], task_train_dataset=ds_task2)

    assert buffer.total_samples == 60
    assert len(buffer.storage[0]) == 20
    assert len(buffer.storage[1]) == 20
    assert len(buffer.storage[2]) == 20


def test_finetune_and_lwf_never_receive_old_class_images_in_training_batch():
    """Verify task training dataloader for finetune/lwf strictly contains only new-task classes."""
    from src.data.tasks.builder import get_task_datasets

    for order in ["A", "B"]:
        tasks = get_task_datasets(order=order, seed=42, image_size=(150, 150))
        for t_idx, t_data in enumerate(tasks, start=1):
            loader = DataLoader(t_data.train_dataset, batch_size=32, shuffle=False)
            batch = next(iter(loader))
            targets = batch[1].tolist()
            # Targets in training batch must strictly belong to t_data.new_classes
            for tgt in targets:
                assert tgt in t_data.new_classes, (
                    f"Leakage detected in Order {order} Task {t_idx}: "
                    f"target {tgt} is not in new_classes {t_data.new_classes}"
                )

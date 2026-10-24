"""Incremental learning experiment runner orchestrating tasks, head expansions, and continual evaluation."""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tasks.builder import get_task_datasets, TASK_ORDERS
from src.evaluation.incremental_metrics.continual import compute_continual_metrics
from src.evaluation.metrics.classification import compute_classification_metrics
from src.incremental.buffers.class_balanced_buffer import ClassBalancedBuffer
from src.incremental.buffers.replay_loader import ReplayDataLoader
from src.incremental.strategies.losses import CrossEntropySeenLoss, LwFLoss
from src.incremental.teachers.manager import create_frozen_teacher
from src.models.custom_cnn.factory import build_model
from src.training.trainers.base_trainer import build_optimizer_and_scheduler, fit
from src.training.trainers.run_context import RunContext
from src.utils.paths import RUNS_DIR
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def evaluate_task_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    seen_classes: List[int],
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Evaluate model on a test loader where model predicts over seen_classes (no task id).

    Args:
        model: Model with output head matching len(seen_classes).
        test_loader: DataLoader supplying test batches.
        seen_classes: Ordered list of global class IDs seen so far.
        device: Active torch device.

    Returns:
        Tuple of (accuracy, all_targets, all_preds_global).
    """
    model.eval()
    all_targets: List[int] = []
    all_preds_global: List[int] = []

    seen_classes_arr = np.array(seen_classes)

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            logits = model(images)
            pred_indices = torch.argmax(logits, dim=1).detach().cpu().numpy()
            pred_globals = seen_classes_arr[pred_indices].tolist()

            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_preds_global.extend(pred_globals)

    acc = (
        float(np.mean(np.array(all_targets) == np.array(all_preds_global)))
        if all_targets
        else 0.0
    )
    return acc, all_targets, all_preds_global


def run_single_incremental_experiment(
    method: str,
    order: str,
    seed: int,
    variant: str = "large",
    epochs_per_task: int = 20,
    device: Optional[torch.device] = None,
    base_runs_dir: Path = RUNS_DIR,
) -> Dict[str, Any]:
    """Execute a complete class-incremental learning sequence (3 tasks).

    Args:
        method: One of ['finetune', 'lwf', 'replay60', 'replay120'].
        order: Task order sequence ('A' or 'B').
        seed: Random seed (42 or 123).
        variant: Architecture variant of Custom CNN (default 'large').
        epochs_per_task: Fixed epoch budget E per task.
        device: Active compute device (CUDA if available, else CPU).
        base_runs_dir: Root directory for saving run artifacts.

    Returns:
        Summary metrics dictionary.
    """
    method = method.lower()
    order = order.upper()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(
        "STARTING INCREMENTAL RUN: method=%s, order=%s, seed=%d, E=%d, device=%s",
        method.upper(),
        order,
        seed,
        epochs_per_task,
        device,
    )
    logger.info("=" * 70)

    # 1. Deterministic seeding
    set_seed(seed, deterministic=True)

    # 2. Initialize Run Context
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ctx = RunContext(
        method=method,
        order=order,
        seed=seed,
        timestamp=f"{variant}_{timestamp}",
        base_runs_dir=base_runs_dir,
    )
    run_ctx.create_tensorboard_writer()
    run_ctx.save_env_info()

    # Save snapshot config
    snapshot_cfg = {
        "method": method,
        "order": order,
        "seed": seed,
        "variant": variant,
        "epochs_per_task": epochs_per_task,
        "device": str(device),
        "task_orders": TASK_ORDERS[order],
    }
    run_ctx.save_config_snapshot(snapshot_cfg)

    # 3. Construct Task Datasets
    tasks_data = get_task_datasets(order=order, seed=seed, image_size=(150, 150))
    num_tasks = len(tasks_data)  # 3 tasks

    # 4. Initialize Model with Task 1 output dimension
    initial_num_classes = len(tasks_data[0].seen_classes)
    model = build_model(variant=variant, num_classes=initial_num_classes).to(device)

    # 5. Initialize Buffer for Experience Replay
    replay_capacity = 0
    if method == "replay60":
        replay_capacity = 60
    elif method == "replay120":
        replay_capacity = 120

    buffer: Optional[ClassBalancedBuffer] = None
    if replay_capacity > 0:
        buffer = ClassBalancedBuffer(capacity=replay_capacity, seed=seed)

    # 6. Initialize Teacher for LwF
    teacher: Optional[nn.Module] = None

    # Track accuracy matrix R[i][j]: accuracy on task j after training task i
    accuracy_matrix: List[List[Optional[float]]] = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]
    per_task_summaries: List[Dict[str, Any]] = []

    total_training_time_sec = 0.0

    # 7. Incremental Task Sequence Loop
    for task_idx in range(1, num_tasks + 1):
        t_data = tasks_data[task_idx - 1]
        seen_classes = t_data.seen_classes
        new_classes = t_data.new_classes
        num_seen = len(seen_classes)
        num_new = len(new_classes)
        num_old = num_seen - num_new

        logger.info("-" * 60)
        logger.info(
            "TASK %d/%d: seen_classes=%s, new_classes=%s (head_dim=%d)",
            task_idx,
            num_tasks,
            seen_classes,
            new_classes,
            num_seen,
        )
        logger.info("-" * 60)

        # 7a. Expand head for Task 2 and Task 3
        if task_idx > 1:
            model.expand(num_new)
            logger.info("Expanded classification head by %d units -> current classes: %d", num_new, model.num_classes)

        # 7b. Learning Rate and Optimizer
        # Task 1 uses 1e-3, Tasks 2 and 3 use 3e-4 as frozen in prompt
        lr = 0.001 if task_idx == 1 else 0.0003
        warmup_epochs = 2 if epochs_per_task >= 10 else 1

        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            lr=lr,
            weight_decay=0.0001,
            num_epochs=epochs_per_task,
            warmup_epochs=warmup_epochs,
            min_lr=1e-6,
        )

        # 7c. DataLoaders
        # Training loader on new task data
        base_train_loader = DataLoader(
            t_data.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        if buffer is not None and buffer.total_samples > 0:
            train_loader = ReplayDataLoader(base_train_loader, buffer=buffer, max_buffer_batch_size=32)
        else:
            train_loader = base_train_loader

        # Validation loader on new classes of current task only (for monitoring only, no tuning)
        val_loader = DataLoader(
            t_data.val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        # 7d. Loss function
        if method == "lwf":
            loss_fn = LwFLoss(
                teacher=teacher,
                seen_classes=seen_classes,
                num_old_classes=num_old,
                temperature=2.0,
                lambda_kd=1.0,
            )
        else:
            loss_fn = CrossEntropySeenLoss(seen_classes=seen_classes)

        # 7e. Train on Task (fixed E epochs, keep last weights, no early stopping)
        t_start = time.time()
        model, _, task_history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            num_epochs=epochs_per_task,
            early_stopping=False,
            restore_best=False,  # strictly retain last epoch's weights
            gradient_clip_val=1.0,
            use_amp=True,
            run_context=run_ctx,
            class_names=[CLASS_NAMES[c] for c in seen_classes],
            verbose=True,
        )
        task_time_sec = time.time() - t_start
        total_training_time_sec += task_time_sec

        # Save per-task checkpoint
        task_ckpt_name = f"checkpoint_task{task_idx}.pt"
        run_ctx.save_checkpoint(
            model,
            filename=task_ckpt_name,
            extra_state={"task_idx": task_idx, "seen_classes": seen_classes},
        )

        # 7f. Update teacher snapshot for LwF after task finishes
        if method == "lwf":
            teacher = create_frozen_teacher(model)

        # 7g. Update replay buffer from TRAIN split only
        if buffer is not None:
            buffer.update_buffer(seen_classes=seen_classes, task_train_dataset=t_data.train_dataset)
            logger.info("Updated Replay Buffer: %d total exemplars across classes %s", buffer.total_samples, buffer.seen_classes)

        # 7h. Official Test Set Evaluation across individual tasks j <= task_idx
        logger.info("Evaluating on official test partitions for all tasks j <= %d...", task_idx)
        for prev_j in range(1, task_idx + 1):
            prev_t_data = tasks_data[prev_j - 1]
            prev_test_ds = prev_t_data.val_dataset.__class__(
                split_csv_path=prev_t_data.cumulative_test_dataset.csv_path,
                split="test",
                allowed_classes=prev_t_data.new_classes,
                transform=prev_t_data.cumulative_test_dataset.transform,
            )
            prev_test_loader = DataLoader(
                prev_test_ds,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
            r_acc, _, _ = evaluate_task_test_set(
                model=model,
                test_loader=prev_test_loader,
                seen_classes=seen_classes,
                device=device,
            )
            accuracy_matrix[task_idx - 1][prev_j - 1] = round(r_acc, 4)

        # 7i. Cumulative Test Evaluation over all seen classes
        cum_test_loader = DataLoader(
            t_data.cumulative_test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        overall_acc, cum_targets, cum_preds = evaluate_task_test_set(
            model=model,
            test_loader=cum_test_loader,
            seen_classes=seen_classes,
            device=device,
        )
        seen_class_names = [CLASS_NAMES[c] for c in seen_classes]
        overall_metrics = compute_classification_metrics(
            cum_targets, cum_preds, class_names=seen_class_names, num_classes=len(seen_classes)
        )

        task_summary = {
            "task_idx": task_idx,
            "seen_classes": seen_classes,
            "new_classes": new_classes,
            "test_accuracy_seen": round(overall_acc, 4),
            "test_macro_f1_seen": round(overall_metrics["macro_f1"], 4),
            "per_class_recall": {cls: overall_metrics["per_class"][cls]["recall"] for cls in seen_class_names},
            "confusion_matrix": overall_metrics["confusion_matrix"],
            "task_time_sec": round(task_time_sec, 2),
            "accuracy_matrix_row": [accuracy_matrix[task_idx - 1][j] for j in range(task_idx)],
        }
        per_task_summaries.append(task_summary)

        logger.info(
            "TASK %d COMPLETED: Test Seen Acc: %.4f, Macro-F1: %.4f | R[%d, :%d] = %s",
            task_idx,
            overall_acc,
            overall_metrics["macro_f1"],
            task_idx,
            task_idx,
            accuracy_matrix[task_idx - 1][:task_idx],
        )

    # 8. Compute Continual Learning Metrics across full matrix R
    continual_metrics = compute_continual_metrics(accuracy_matrix, memory_cost=replay_capacity)

    # Final summary dictionary
    final_metrics = {
        "method": method,
        "order": order,
        "seed": seed,
        "variant": variant,
        "epochs_per_task": epochs_per_task,
        "total_training_time_sec": round(total_training_time_sec, 2),
        "replay_memory_cost": replay_capacity,
        # Matrix R and continual indices
        "accuracy_matrix": accuracy_matrix,
        "task_average_accuracies": continual_metrics["task_average_accuracies"],
        "average_incremental_accuracy": continual_metrics["average_incremental_accuracy"],
        "task_forgetting": continual_metrics["task_forgetting"],
        "overall_forgetting": continual_metrics["overall_forgetting"],
        "backward_transfer": continual_metrics["backward_transfer"],
        # Final Task 3 full test metrics
        "final_accuracy": per_task_summaries[-1]["test_accuracy_seen"],
        "final_macro_f1": per_task_summaries[-1]["test_macro_f1_seen"],
        "final_per_class_recall": per_task_summaries[-1]["per_class_recall"],
        "final_confusion_matrix": per_task_summaries[-1]["confusion_matrix"],
        "per_task_summaries": per_task_summaries,
    }

    # Save to metrics.json
    run_ctx.save_metrics(final_metrics)
    run_ctx.close()

    logger.info("=" * 70)
    logger.info(
        "FINISHED INCREMENTAL RUN: %s (Order %s, Seed %d) | A_avg: %.4f, Forgetting: %.4f, BWT: %.4f, Final F1: %.4f",
        method.upper(),
        order,
        seed,
        continual_metrics["average_incremental_accuracy"],
        continual_metrics["overall_forgetting"],
        continual_metrics["backward_transfer"],
        final_metrics["final_macro_f1"],
    )
    logger.info("=" * 70)

    return final_metrics

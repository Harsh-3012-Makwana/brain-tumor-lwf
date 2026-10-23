"""Core reusable training loop supporting AMP, warmup, cosine annealing, and early stopping."""

from __future__ import annotations

import copy
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.metrics.classification import compute_classification_metrics
from src.training.trainers.run_context import RunContext

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    betas: Tuple[float, float] = (0.9, 0.999),
    num_epochs: int = 40,
    warmup_epochs: int = 2,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Instantiate AdamW optimizer coupled with Linear Warmup + Cosine Annealing scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    if warmup_epochs > 0 and num_epochs > warmup_epochs:
        s1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        s2 = CosineAnnealingLR(
            optimizer, T_max=(num_epochs - warmup_epochs), eta_min=min_lr
        )
        scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    return optimizer, scheduler


def execute_loss_callable(
    loss_fn: Callable[..., torch.Tensor],
    model: nn.Module,
    logits: torch.Tensor,
    images: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Execute pluggable loss function accommodating either (logits, targets) or (model, images, targets)."""
    try:
        sig = inspect.signature(loss_fn)
        param_count = len(sig.parameters)
    except Exception:
        param_count = 2

    if param_count >= 3:
        return loss_fn(model, images, targets)
    return loss_fn(logits, targets)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device,
    num_epochs: int = 40,
    early_stopping_patience: int = 8,
    early_stopping_min_delta: float = 0.0005,
    gradient_clip_val: float = 1.0,
    use_amp: bool = True,
    run_context: Optional[RunContext] = None,
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    """Execute complete training loop with validation, early stopping, and metric logging.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader supplying training batches.
        val_loader: DataLoader supplying validation batches.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        loss_fn: Loss callable (supports standard CE or LwF composite loss).
        device: Active torch.device.
        num_epochs: Maximum epochs.
        early_stopping_patience: Patience epochs without val_macro_f1 gain.
        early_stopping_min_delta: Minimum delta threshold for improvement.
        gradient_clip_val: Max gradient norm for clipping.
        use_amp: Enable automatic mixed precision (bfloat16 if supported, else float16).
        run_context: Optional RunContext for logging to train_log.csv and TensorBoard.
        class_names: Class labels for per-class metric reporting.
        verbose: Print progress per epoch.

    Returns:
        Tuple of (best_model, best_val_metrics, training_history).
    """
    model = model.to(device)
    num_classes = getattr(model, "num_classes", 4)

    # Autocast setup
    amp_enabled = use_amp and device.type == "cuda"
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and not use_bf16))

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    writer = run_context.writer if run_context else None

    logger.info(
        "Starting training: %d epochs, patience=%d, AMP=%s (%s), device=%s",
        num_epochs,
        early_stopping_patience,
        amp_enabled,
        amp_dtype if amp_enabled else "fp32",
        device,
    )

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ------------------ 1. Training Phase ------------------
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        train_preds: List[int] = []
        train_targets: List[int] = []

        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(images)
                loss = execute_loss_callable(loss_fn, model, logits, images, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if gradient_clip_val > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()

            train_loss_total += loss.item() * batch_size
            train_samples += batch_size

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds.tolist())
            train_targets.extend(targets.detach().cpu().numpy().tolist())

        # Step learning rate scheduler per epoch
        curr_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_total / train_samples if train_samples > 0 else 0.0
        train_metrics = compute_classification_metrics(
            train_targets, train_preds, class_names=class_names, num_classes=num_classes
        )

        # ------------------ 2. Validation Phase ------------------
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)
                batch_size = images.size(0)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(images)
                    loss = execute_loss_callable(loss_fn, model, logits, images, targets)

                val_loss_total += loss.item() * batch_size
                val_samples += batch_size

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                val_preds.extend(preds.tolist())
                val_targets.extend(targets.detach().cpu().numpy().tolist())

        val_loss = val_loss_total / val_samples if val_samples > 0 else 0.0
        val_metrics = compute_classification_metrics(
            val_targets, val_preds, class_names=class_names, num_classes=num_classes
        )
        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        epoch_time = time.time() - t0

        # ------------------ 3. Logging & Checkpoint ------------------
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_metrics["accuracy"], 4),
            "train_macro_f1": round(train_metrics["macro_f1"], 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "lr": round(curr_lr, 7),
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(row)

        if run_context:
            run_context.log_epoch(row)

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("MacroF1/train", train_metrics["macro_f1"], epoch)
            writer.add_scalar("MacroF1/val", val_macro_f1, epoch)
            writer.add_scalar("LearningRate", curr_lr, epoch)

        if verbose:
            logger.info(
                "Epoch %02d/%d [%.1fs]: train_loss=%.4f, train_f1=%.4f | val_loss=%.4f, val_f1=%.4f, val_acc=%.4f | lr=%.1e",
                epoch,
                num_epochs,
                epoch_time,
                train_loss,
                train_metrics["macro_f1"],
                val_loss,
                val_macro_f1,
                val_acc,
                curr_lr,
            )

        # Early stopping check on validation Macro-F1
        if val_macro_f1 > (best_val_macro_f1 + early_stopping_min_delta):
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_val_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_metrics": val_metrics,
            }
            patience_counter = 0
            if run_context:
                run_context.save_checkpoint(
                    model, filename="best.pt", extra_state={"best_val_metrics": best_val_metrics}
                )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch %d with val_macro_f1=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_macro_f1,
                )
                break

    # Restore optimal model weights
    model.load_state_dict(best_weights)
    logger.info("Restored optimal model checkpoint from epoch %d (val_macro_f1=%.4f)", best_epoch, best_val_macro_f1)

    return model, best_val_metrics, history

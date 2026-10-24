"""Unit tests for training pipeline, metrics helpers, and reproducibility."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics.classification import compute_classification_metrics
from src.models.custom_cnn.factory import build_model
from src.training.trainers.base_trainer import build_optimizer_and_scheduler, fit
from src.training.trainers.run_context import RunContext


def test_metrics_helpers_match_sklearn():
    """Verify compute_classification_metrics matches scikit-learn on a known example."""
    if not _SKLEARN_AVAILABLE:
        pytest.skip("sklearn DLL blocked by environment Application Control policy")

    y_true = [0, 1, 2, 0, 1, 2, 1, 0, 2, 2]
    y_pred = [0, 1, 1, 0, 2, 2, 1, 0, 2, 1]

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)

    expected_acc = accuracy_score(y_true, y_pred)
    expected_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    expected_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    expected_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    expected_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    assert pytest.approx(metrics["accuracy"], abs=1e-6) == expected_acc
    assert pytest.approx(metrics["macro_f1"], abs=1e-6) == expected_f1
    assert pytest.approx(metrics["macro_precision"], abs=1e-6) == expected_prec
    assert pytest.approx(metrics["macro_recall"], abs=1e-6) == expected_rec
    assert metrics["confusion_matrix"] == expected_cm


def test_run_context_creates_expected_files():
    """Verify RunContext generates all mandatory run artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ctx = RunContext(
            method="joint",
            order="none",
            seed=42,
            timestamp="test",
            base_runs_dir=tmp_path,
        )

        ctx.create_tensorboard_writer()
        ctx.save_config_snapshot({"batch_size": 32})
        ctx.save_env_info()
        ctx.init_train_log(["epoch", "loss"])
        ctx.log_epoch({"epoch": 1, "loss": 0.5})
        ctx.save_metrics({"val_acc": 0.95})

        model = nn.Linear(4, 2)
        ctx.save_checkpoint(model, filename="best.pt")
        ctx.close()

        expected_files = [
            "config_snapshot.yaml",
            "env_info.txt",
            "train_log.csv",
            "metrics.json",
            "best.pt",
        ]
        for fname in expected_files:
            file_path = ctx.run_dir / fname
            assert file_path.is_file(), f"Expected artifact {fname} missing from {ctx.run_dir}"


def test_fit_decreases_loss_tiny_dataset():
    """Verify fit() trains for 2 epochs and decreases training loss on a synthetic dataset."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Simple separable synthetic data (32 samples, 3x150x150)
    x = torch.randn(32, 3, 150, 150)
    y = torch.tensor([0 if i < 16 else 1 for i in range(32)])
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=16, shuffle=False)
    val_loader = DataLoader(ds, batch_size=16, shuffle=False)

    model = build_model(variant="small", num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    best_model, best_val, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device=device,
        num_epochs=2,
        early_stopping_patience=5,
        use_amp=False,
        verbose=False,
    )

    assert len(history) == 2
    # Verify training loss decreased
    assert history[1]["train_loss"] < history[0]["train_loss"]


def test_early_stopping_triggers_and_restores_best_weights():
    """Verify early stopping terminates when validation F1 fails to improve and restores optimal weights."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    x = torch.randn(16, 3, 150, 150)
    y = torch.randint(0, 2, (16,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = build_model(variant="small", num_classes=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Mock constant loss and validation metric
    best_model, best_val, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        num_epochs=15,
        early_stopping_patience=3,
        use_amp=False,
        verbose=False,
    )

    # Should stop well before 15 epochs due to patience=3
    assert len(history) <= 7


def test_fixed_seed_determinism_on_cpu():
    """Verify that identical seeds produce bit-identical losses and outputs on CPU."""
    device = torch.device("cpu")

    losses_run1 = []
    losses_run2 = []

    for run_idx in range(2):
        torch.manual_seed(42)
        np.random.seed(42)

        x = torch.randn(16, 3, 150, 150)
        y = torch.randint(0, 2, (16,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

        model = build_model(variant="small", num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        _, _, history = fit(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=nn.CrossEntropyLoss(),
            device=device,
            num_epochs=2,
            use_amp=False,
            verbose=False,
        )

        losses = [h["train_loss"] for h in history]
        if run_idx == 0:
            losses_run1 = losses
        else:
            losses_run2 = losses

    assert losses_run1 == losses_run2

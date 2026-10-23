"""Training and validation loss and accuracy curve visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def plot_training_curves(
    history: List[Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = "Training and Validation Progress",
) -> None:
    """Plot dual-axis or subplots of loss and accuracy/F1 over training epochs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#d62728", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy / F1 plot
    ax2.plot(epochs, train_acc, label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#ff7f0e", lw=2, linestyle="--")
    ax2.plot(epochs, val_f1, label="Val Macro-F1", color="#9467bd", lw=2, linestyle="-.")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score (0 - 1.0)", fontsize=10)
    ax2.set_title("Accuracy & Macro-F1 vs. Epochs", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

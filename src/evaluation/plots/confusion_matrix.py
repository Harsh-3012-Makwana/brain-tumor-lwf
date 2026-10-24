"""Confusion matrix visualization utility using pure Matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    output_path: Path,
    title: Optional[str] = "Normalized Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and save confusion matrix heatmap using pure Matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm, dtype=float)

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_arr / row_sums
    else:
        cm_display = cm_arr

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([c.capitalize() for c in class_names], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([c.capitalize() for c in class_names], fontsize=9)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            text = f"{val:.1%}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    ax.set_xlabel("Predicted Class", fontsize=10, labelpad=8)
    ax.set_ylabel("True Ground Truth Class", fontsize=10, labelpad=8)
    if title:
        ax.set_title(title, fontsize=11, pad=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

"""Confusion matrix visualization utility using Matplotlib and Seaborn."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    output_path: Path,
    title: Optional[str] = "Normalized Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and save confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm, dtype=float)

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        # Avoid zero division
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_arr / row_sums
        fmt = ".2%"
    else:
        cm_display = cm_arr
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_xlabel("Predicted Class", fontsize=10, labelpad=8)
    ax.set_ylabel("True Ground Truth Class", fontsize=10, labelpad=8)
    if title:
        ax.set_title(title, fontsize=11, pad=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

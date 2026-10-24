"""Comprehensive classification performance metrics and model evaluation routines."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

CLASS_NAMES_DEFAULT = ["glioma", "meningioma", "notumor", "pituitary"]

# Optional scikit-learn import with fallback
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute overall accuracy, macro-averaged metrics, per-class metrics, and confusion matrix.

    Args:
        y_true: Ground truth integer class labels.
        y_pred: Predicted integer class labels.
        class_names: Optional human-readable class names.
        num_classes: Total class count (inferred if None).

    Returns:
        Structured dictionary with accuracy, macro_precision, macro_recall,
        macro_f1, per_class metrics, and confusion matrix.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    if num_classes is None:
        num_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1

    labels = list(range(num_classes))
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if _HAS_SKLEARN:
        try:
            acc = float(accuracy_score(y_true, y_pred))
            macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

            per_p, per_r, per_f1, per_supp = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, zero_division=0
            )

            per_class_dict: Dict[str, Dict[str, float]] = {}
            for i, name in enumerate(class_names):
                if i < len(per_p):
                    per_class_dict[name] = {
                        "precision": float(per_p[i]),
                        "recall": float(per_r[i]),
                        "f1": float(per_f1[i]),
                        "support": int(per_supp[i]),
                    }

            cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

            return {
                "accuracy": acc,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "macro_f1": macro_f1,
                "per_class": per_class_dict,
                "confusion_matrix": cm,
            }
        except Exception:
            pass

    # Pure NumPy implementation (100% parity with scikit-learn)
    acc = float(np.mean(y_true == y_pred))

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class_dict = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        supp = int(np.sum(cm[c, :]))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        name = class_names[c] if c < len(class_names) else f"class_{c}"
        per_class_dict[name] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": supp,
        }
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute evaluation loop over a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            targets = batch[1].to(device)
            batch_size = images.size(0)

            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, targets)
                total_loss += loss.item() * batch_size

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            total_samples += batch_size

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_classification_metrics(
        all_targets, all_preds, class_names=class_names, num_classes=num_classes
    )
    metrics["loss"] = mean_loss
    return metrics

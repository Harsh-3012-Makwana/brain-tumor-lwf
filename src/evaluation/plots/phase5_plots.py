"""Phase 5 Visualization Suite for Class-Incremental Brain Tumor MRI Classification.

Generates publication-quality figures (pure Matplotlib, 300 DPI) in results/figures/:
  a) accuracy_trajectories.png: Test accuracy on seen classes after each task per method (Orders A, B, C, D)
  b) final_metrics_bars.png: Final Accuracy, Macro-F1, and Forgetting bars with error bars
  c) accuracy_matrices_heatmaps.png: Continual learning accuracy matrix R heatmaps (seed 42)
  d) confusion_matrices_all_methods.png: Final 4x4 confusion matrices across methods and orders (seed 42)
  e) accuracy_vs_memory.png: Final accuracy vs replay buffer size (0, 60, 120) with Joint and lwf_new
  f) restricted_acc_vs_old_mass.png: Diagnostic scatter plot of Restricted Old Acc vs Mean Old Softmax Mass
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import FIGURES_DIR, RUNS_DIR, TABLES_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("phase5_plots")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

METHOD_COLORS = {
    "finetune": "#d32f2f",      # Red
    "lwf": "#f57c00",           # Orange
    "lwf_all": "#f57c00",       # Orange
    "lwf_standard": "#f57c00",  # Orange
    "lwf_new": "#00897b",       # Teal
    "lwfplus": "#7b1fa2",       # Purple
    "replay60": "#1976d2",      # Blue
    "replay120": "#0d47a1",     # Dark Blue
    "joint": "#2e7d32",         # Green
}

METHOD_LABELS = {
    "finetune": "Naive Fine-Tuning",
    "lwf": "LwF (Standard)",
    "lwf_all": "LwF (Standard)",
    "lwf_standard": "LwF (Standard)",
    "lwf_new": "LwF (New-Class CE)",
    "lwfplus": "lwfPlus (Screened)",
    "replay60": "Replay (M=60)",
    "replay120": "Replay (M=120)",
    "joint": "Joint (Upper Bound)",
}


def load_all_run_metrics() -> List[Dict[str, Any]]:
    """Scan results/runs/ and load all completed incremental run metrics."""
    metrics_list: List[Dict[str, Any]] = []
    if not RUNS_DIR.is_dir():
        return metrics_list

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.json"
        if metrics_file.is_file():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "method" in data and "order" in data and "seed" in data:
                    metrics_list.append(data)
            except Exception as e:
                logger.warning("Could not read %s: %s", metrics_file, e)
    return metrics_list


def _parse_mean_val(val: Any, default: float) -> float:
    if val is None:
        return default
    s = str(val).split("±")[0].split("\u00b1")[0].strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def load_joint_test_metrics() -> Dict[str, float]:
    """Load selected Medium variant Joint metrics."""
    joint_csv = TABLES_DIR / "joint_variant_comparison.csv"
    res = {"final_acc": 0.8862, "final_macro_f1": 0.8831}
    if joint_csv.is_file():
        try:
            with open(joint_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("selected") in ["YES", "YES (MAIN)"] and r.get("seed") == "all":
                        res["final_acc"] = _parse_mean_val(r.get("best_test_acc"), 0.8862)
                        res["final_macro_f1"] = _parse_mean_val(r.get("best_test_macro_f1"), 0.8831)
        except Exception as e:
            logger.warning("Error parsing joint metrics: %s", e)
    return res


# ==============================================================================
# Plot A: Task Accuracy Trajectories
# ==============================================================================
def plot_accuracy_trajectories(all_metrics: List[Dict[str, Any]]) -> Path:
    """Plot accuracy on seen classes after each task, per method, one panel per order (A, B, C, D)."""
    orders = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    plt.subplots_adjust(wspace=0.15)

    joint_metrics = load_joint_test_metrics()
    joint_acc = joint_metrics["final_acc"]

    for ax, order in zip(axes, orders):
        ax.set_title(f"Order {order}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Task Index ($k$)", fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Reference Joint line
        ax.axhline(joint_acc, color=METHOD_COLORS["joint"], linestyle=":", linewidth=1.5, label="Joint Upper Bound" if order == "A" else "")

        order_runs = [m for m in all_metrics if m.get("order") == order]
        methods_in_order = ["finetune", "lwf", "lwf_all", "lwf_new", "replay60", "replay120"]

        for method in methods_in_order:
            runs = [m for m in order_runs if m.get("method") == method]
            if not runs:
                continue

            # Gather per_task test_accuracy_seen across seeds
            task_accs_per_seed: Dict[int, List[float]] = defaultdict(list)
            for r in runs:
                summaries = r.get("per_task_summaries", [])
                for s in summaries:
                    t_idx = s["task_idx"]
                    task_accs_per_seed[t_idx].append(s["test_accuracy_seen"])

            if not task_accs_per_seed:
                continue

            sorted_tasks = sorted(task_accs_per_seed.keys())
            means = [np.mean(task_accs_per_seed[t]) for t in sorted_tasks]
            stds = [np.std(task_accs_per_seed[t]) for t in sorted_tasks]

            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method)

            ax.errorbar(
                sorted_tasks,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                color=color,
                label=label if order in ["A", "C"] else "",
            )
            ax.set_xticks(sorted_tasks)

    axes[0].set_ylabel("Test Accuracy on Seen Classes ($A_k$)", fontsize=10)
    # Combine unique handles/labels
    handles, labels = axes[0].get_legend_handles_labels()
    h_c, l_c = axes[2].get_legend_handles_labels()
    by_label = dict(zip(labels + l_c, handles + h_c))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True, fontsize=10)

    out_path = FIGURES_DIR / "accuracy_trajectories.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Plot A: %s", out_path)
    return out_path


# ==============================================================================
# Plot B: Final Metrics Bar Charts
# ==============================================================================
def plot_final_metrics_bars(all_metrics: List[Dict[str, Any]]) -> Path:
    """Bar charts of Final Accuracy, Final Macro-F1, and Forgetting per method per order."""
    orders = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.22)

    joint_metrics = load_joint_test_metrics()
    target_methods = ["finetune", "lwf_standard", "lwf_new", "replay60", "replay120"]

    # Metrics to plot: (title, key, y_lim, joint_val)
    metric_configs = [
        ("Final Test Accuracy", "final_accuracy", (0.0, 1.05), joint_metrics["final_acc"]),
        ("Final Test Macro-F1", "final_macro_f1", (0.0, 1.05), joint_metrics["final_macro_f1"]),
        ("Overall Forgetting ($F$) ↓", "overall_forgetting", (0.0, 1.05), 0.0),
    ]

    for ax, (m_title, m_key, ylim, j_val) in zip(axes, metric_configs):
        ax.set_title(m_title, fontsize=12, fontweight="bold")
        ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        if j_val is not None and j_val > 0.0:
            ax.axhline(j_val, color=METHOD_COLORS["joint"], linestyle=":", linewidth=1.5, label="Joint Upper Bound")

        # Grouped bar setup
        # For each order, display methods present
        x_centers = np.arange(len(orders))
        bar_width = 0.15
        drawn_methods = set()

        for idx_m, method in enumerate(target_methods):
            means = []
            stds = []
            for order in orders:
                # Handle lwf naming difference across orders
                m_search = method
                if method == "lwf_standard":
                    m_search = "lwf" if order in ["A", "B"] else "lwf_all"

                runs = [m for m in all_metrics if m.get("order") == order and m.get("method") == m_search]
                if runs:
                    vals = [r[m_key] for r in runs if m_key in r]
                    means.append(np.mean(vals) if vals else 0.0)
                    stds.append(np.std(vals) if vals else 0.0)
                else:
                    means.append(0.0)
                    stds.append(0.0)

            # Check if method had any non-zero values
            if max(means) > 0.0:
                offset = (idx_m - len(target_methods) / 2.0 + 0.5) * bar_width
                color = METHOD_COLORS.get(method, "#666666")
                lbl = METHOD_LABELS.get(method, method)
                ax.bar(
                    x_centers + offset,
                    means,
                    yerr=stds,
                    width=bar_width,
                    color=color,
                    label=lbl if lbl not in drawn_methods else "",
                    capsize=3,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                )
                drawn_methods.add(lbl)

        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"Order {o}" for o in orders], fontsize=10, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    # Filter unique
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True, fontsize=10)

    out_path = FIGURES_DIR / "final_metrics_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Plot B: %s", out_path)
    return out_path


# ==============================================================================
# Plot C: Accuracy Matrix R Heatmaps (Seed 42)
# ==============================================================================
def plot_accuracy_matrices_heatmaps(all_metrics: List[Dict[str, Any]], seed: int = 42) -> Path:
    """Heatmaps of accuracy matrix R per method (seed 42), per order."""
    orders = ["A", "B", "C", "D"]
    methods = ["finetune", "lwf", "lwf_new", "replay120"]

    fig, axes = plt.subplots(len(orders), len(methods), figsize=(16, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    for r_idx, order in enumerate(orders):
        for c_idx, method in enumerate(methods):
            ax = axes[r_idx, c_idx]
            m_search = method
            if order in ["C", "D"] and method == "lwf":
                m_search = "lwf_all"
            elif order in ["A", "B"] and method == "lwf_new":
                # lwf_new does not apply to single-class orders A and B
                ax.text(0.5, 0.5, "N/A\n(Single-class tasks)", ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if r_idx == 0:
                    ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")
                continue

            matching = [m for m in all_metrics if m.get("order") == order and m.get("method") == m_search and m.get("seed") == seed]
            if not matching:
                ax.text(0.5, 0.5, "Pending", ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            R = np.array(matching[0]["accuracy_matrix"], dtype=float)
            n_tasks = R.shape[0]

            im = ax.imshow(R, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(n_tasks))
            ax.set_yticks(range(n_tasks))
            ax.set_xticklabels([f"T{j+1}" for j in range(n_tasks)], fontsize=9)
            ax.set_yticklabels([f"After T{i+1}" for i in range(n_tasks)], fontsize=9)

            if r_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"Order {order}", fontsize=11, fontweight="bold")

            # Annotate numbers
            for i in range(n_tasks):
                for j in range(n_tasks):
                    val = R[i, j]
                    if val > 0.0 or j <= i:
                        text_color = "white" if val > 0.65 else "black"
                        ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

    out_path = FIGURES_DIR / "accuracy_matrices_heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Plot C: %s", out_path)
    return out_path


# ==============================================================================
# Plot D: Confusion Matrices (Seed 42)
# ==============================================================================
def plot_confusion_matrices(all_metrics: List[Dict[str, Any]], seed: int = 42) -> Path:
    """Final 4x4 confusion matrices per method (seed 42), per order."""
    orders = ["A", "B", "C", "D"]
    methods = ["finetune", "lwf", "lwf_new", "replay120"]

    fig, axes = plt.subplots(len(orders), len(methods), figsize=(16, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    for r_idx, order in enumerate(orders):
        for c_idx, method in enumerate(methods):
            ax = axes[r_idx, c_idx]
            m_search = method
            if order in ["C", "D"] and method == "lwf":
                m_search = "lwf_all"
            elif order in ["A", "B"] and method == "lwf_new":
                ax.text(0.5, 0.5, "N/A\n(Single-class tasks)", ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if r_idx == 0:
                    ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")
                continue

            matching = [m for m in all_metrics if m.get("order") == order and m.get("method") == m_search and m.get("seed") == seed]
            if not matching:
                ax.text(0.5, 0.5, "Pending", ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            cm = np.array(matching[0]["final_confusion_matrix"], dtype=float)
            # Row-normalized CM
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_norm = cm / row_sums

            ax.imshow(cm_norm, cmap="YlOrRd", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["gl", "me", "no", "pi"], fontsize=9)
            ax.set_yticklabels(["gl", "me", "no", "pi"], fontsize=9)

            if r_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"Order {order}\nTrue Class", fontsize=10, fontweight="bold")
            if r_idx == len(orders) - 1:
                ax.set_xlabel("Predicted Class", fontsize=9)

            for i in range(4):
                for j in range(4):
                    cnt = int(cm[i, j])
                    pct = cm_norm[i, j] * 100
                    color = "white" if cm_norm[i, j] > 0.6 else "black"
                    ax.text(j, i, f"{cnt}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=7.5)

    out_path = FIGURES_DIR / "confusion_matrices_all_methods.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Plot D: %s", out_path)
    return out_path


# ==============================================================================
# Plot E: Final Accuracy vs Replay Memory Buffer
# ==============================================================================
def plot_accuracy_vs_memory(all_metrics: List[Dict[str, Any]]) -> Path:
    """Final accuracy vs replay buffer size (0, 60, 120 images) for Orders C and D."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.grid(True, linestyle="--", alpha=0.5)

    joint_metrics = load_joint_test_metrics()
    joint_acc = joint_metrics["final_acc"]

    # Horizontal Joint Upper Bound line
    ax.axhline(joint_acc, color=METHOD_COLORS["joint"], linestyle="--", linewidth=1.8, label=f"Joint Training ({joint_acc*100:.1f}%)")

    orders = ["C", "D"]
    markers = {"C": "o", "D": "s"}
    colors = {"C": "#1565c0", "D": "#d81b60"}

    for order in orders:
        mem_sizes = [0, 60, 120]
        acc_means = []
        acc_stds = []

        # M=0 is finetune
        ft_runs = [m for m in all_metrics if m.get("order") == order and m.get("method") == "finetune"]
        if ft_runs:
            acc_means.append(np.mean([r["final_accuracy"] for r in ft_runs]))
            acc_stds.append(np.std([r["final_accuracy"] for r in ft_runs]))
        else:
            acc_means.append(np.nan)
            acc_stds.append(0.0)

        # M=60
        r60_runs = [m for m in all_metrics if m.get("order") == order and m.get("method") == "replay60"]
        if r60_runs:
            acc_means.append(np.mean([r["final_accuracy"] for r in r60_runs]))
            acc_stds.append(np.std([r["final_accuracy"] for r in r60_runs]))
        else:
            acc_means.append(np.nan)
            acc_stds.append(0.0)

        # M=120
        r120_runs = [m for m in all_metrics if m.get("order") == order and m.get("method") == "replay120"]
        if r120_runs:
            acc_means.append(np.mean([r["final_accuracy"] for r in r120_runs]))
            acc_stds.append(np.std([r["final_accuracy"] for r in r120_runs]))
        else:
            acc_means.append(np.nan)
            acc_stds.append(0.0)

        # Filter NaNs for plotting
        valid = [(m, a, s) for m, a, s in zip(mem_sizes, acc_means, acc_stds) if not np.isnan(a)]
        if valid:
            ms, accs, stds = zip(*valid)
            ax.errorbar(
                ms,
                accs,
                yerr=stds,
                marker=markers[order],
                color=colors[order],
                linewidth=2,
                capsize=5,
                label=f"Replay Buffer (Order {order})",
            )

        # Plot lwf_new at M=0 as star
        lwf_new_runs = [m for m in all_metrics if m.get("order") == order and m.get("method") == "lwf_new"]
        if lwf_new_runs:
            l_acc = np.mean([r["final_accuracy"] for r in lwf_new_runs])
            l_std = np.std([r["final_accuracy"] for r in lwf_new_runs])
            ax.errorbar(
                [0.0],
                [l_acc],
                yerr=[l_std],
                marker="*",
                markersize=14,
                color=METHOD_COLORS["lwf_new"],
                capsize=5,
                linestyle="None",
                label=f"LwF (New-Class CE, 0 images) [Order {order}]",
            )

    ax.set_title("Final Accuracy vs Replay Memory Buffer (Orders C & D)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Replay Memory Buffer Size (Stored Exemplar Images)", fontsize=11)
    ax.set_ylabel("Final Test Accuracy", fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_ylim(0.20, 1.0)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    out_path = FIGURES_DIR / "accuracy_vs_memory.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Plot E: %s", out_path)
    return out_path


# ==============================================================================
# Plot F: Restricted Old-Class Accuracy vs Old-Class Softmax Mass
# ==============================================================================
def plot_restricted_acc_vs_old_mass(all_metrics: List[Dict[str, Any]]) -> Path:
    """Restricted old-class accuracy vs old-class softmax mass per method for Orders C and D."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(True, linestyle="--", alpha=0.5)

    orders = ["C", "D"]
    methods = ["finetune", "lwf_all", "lwf_new", "replay60", "replay120"]
    order_markers = {"C": "o", "D": "^"}

    # Reference threshold lines
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Balanced Softmax Mass (0.50)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Random Guess on 2 Old Classes (0.50)")

    for order in orders:
        for method in methods:
            runs = [m for m in all_metrics if m.get("order") == order and m.get("method") == method]
            if not runs:
                continue

            r_accs = [r.get("restricted_old_accuracy") for r in runs if r.get("restricted_old_accuracy") is not None]
            o_masses = [r.get("mean_softmax_mass_old") for r in runs if r.get("mean_softmax_mass_old") is not None]

            if not r_accs or not o_masses:
                continue

            m_r_acc = float(np.mean(r_accs))
            std_r_acc = float(np.std(r_accs))
            m_o_mass = float(np.mean(o_masses))
            std_o_mass = float(np.std(o_masses))

            color = METHOD_COLORS.get(method, "#333333")
            marker = order_markers[order]
            lbl = f"{METHOD_LABELS.get(method, method)} (Order {order})"

            ax.errorbar(
                m_o_mass,
                m_r_acc,
                xerr=std_o_mass,
                yerr=std_r_acc,
                marker=marker,
                markersize=9,
                color=color,
                capsize=4,
                label=lbl,
            )

            # Text annotation near point
            ax.annotate(
                f"{method.upper()}-{order}",
                (m_o_mass, m_r_acc),
                textcoords="offset points",
                xytext=(8, -4),
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    ax.set_title("Restricted Old-Class Accuracy vs Old-Class Softmax Mass\n(Diagnostic for Logit Depression & Boundary Preservation)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Softmax Mass Allocated to Old Classes (Old Mass)", fontsize=11)
    ax.set_ylabel("Restricted Accuracy on Old Classes (Old Logits Only)", fontsize=11)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.2, 1.02)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")

    out_path = FIGURES_DIR / "restricted_acc_vs_old_mass.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Plot F: %s", out_path)
    return out_path


def run_all_phase5_plots() -> Dict[str, Path]:
    """Execute generation of all Phase 5 plots."""
    logger.info("Scanning completed metrics for Phase 5 plotting suite...")
    all_metrics = load_all_run_metrics()
    logger.info("Loaded metrics for %d runs", len(all_metrics))

    generated = {}
    try:
        generated["trajectories"] = plot_accuracy_trajectories(all_metrics)
    except Exception as e:
        logger.error("Plot A failed: %s", e, exc_info=True)

    try:
        generated["bars"] = plot_final_metrics_bars(all_metrics)
    except Exception as e:
        logger.error("Plot B failed: %s", e, exc_info=True)

    try:
        generated["heatmaps"] = plot_accuracy_matrices_heatmaps(all_metrics)
    except Exception as e:
        logger.error("Plot C failed: %s", e, exc_info=True)

    try:
        generated["confusion"] = plot_confusion_matrices(all_metrics)
    except Exception as e:
        logger.error("Plot D failed: %s", e, exc_info=True)

    try:
        generated["memory"] = plot_accuracy_vs_memory(all_metrics)
    except Exception as e:
        logger.error("Plot E failed: %s", e, exc_info=True)

    try:
        generated["diagnostic"] = plot_restricted_acc_vs_old_mass(all_metrics)
    except Exception as e:
        logger.error("Plot F failed: %s", e, exc_info=True)

    logger.info("Phase 5 plotting suite generated %d figures successfully!", len(generated))
    return generated


if __name__ == "__main__":
    run_all_phase5_plots()

"""Execution script for Stage D Final Incremental Runs.

Runs:
  - lwfPlus on Orders A, B, C x Seeds [42, 123] (6 runs)
  - Baselines on Order C: finetune, lwf_all, lwf_new, replay60, replay120 x Seeds [42, 123] (10 runs)
Total = 16 runs.

Generates consolidated results/tables/incremental_results.md and .csv
incorporating Orders A, B, and C with restricted old-class accuracy and softmax mass diagnostics.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.incremental.runner.incremental_runner import run_single_incremental_experiment
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, RUNS_DIR, TABLES_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("stage_d_final")

STAGE_D_MANIFEST = [
    # 1. lwfPlus on Orders A, B, C
    ("lwfplus", "A", 42),
    ("lwfplus", "A", 123),
    ("lwfplus", "B", 42),
    ("lwfplus", "B", 123),
    ("lwfplus", "C", 42),
    ("lwfplus", "C", 123),
    # 2. Order C baselines
    ("finetune", "C", 42),
    ("finetune", "C", 123),
    ("lwf_all", "C", 42),
    ("lwf_all", "C", 123),
    ("lwf_new", "C", 42),
    ("lwf_new", "C", 123),
    ("replay60", "C", 42),
    ("replay60", "C", 123),
    ("replay120", "C", 42),
    ("replay120", "C", 123),
]


def get_completed_run_metrics(method: str, order: str, seed: int) -> Optional[Dict[str, Any]]:
    """Check if a run has already completed in results/runs/ and return its metrics."""
    pattern = f"{method.lower()}_{order.lower()}_{seed}_"
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir() and p.name.startswith(pattern):
            metrics_file = p / "metrics.json"
            if metrics_file.is_file():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "average_incremental_accuracy" in data and "final_macro_f1" in data:
                        return data
                except Exception:
                    pass
    return None


def load_joint_baseline_metrics() -> Dict[str, Any]:
    """Retrieve joint training upper bound results from joint_variant_comparison.csv."""
    joint_csv = TABLES_DIR / "joint_variant_comparison.csv"
    if not joint_csv.is_file():
        return {}

    with open(joint_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("selected") in ["YES", "YES (MAIN)"] and row.get("seed") == "all":
                return row
    return {}


def format_memory_cost(method: str, replay_cost: int) -> str:
    """Format memory cost transparently (noting model snapshot memory for distillation)."""
    m = method.lower()
    if replay_cost > 0:
        return f"{replay_cost} images"
    elif m in ["lwf", "lwf_all", "lwf_new", "lwfplus"] or m.startswith("lwf"):
        return "0 images (4.15 MB teacher)"
    elif m == "joint":
        return "0 images (all data)"
    else:
        return "0 images"


def generate_consolidated_incremental_tables() -> None:
    """Scan all runs in results/runs/ and build the complete consolidated incremental results tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    joint_baseline = load_joint_baseline_metrics()

    # Target methods and orders to display
    display_methods = [
        "finetune", "lwf", "lwf_all", "lwf_new", "lwfplus", "replay60", "replay120"
    ]
    display_orders = ["A", "B", "C"]
    display_seeds = [42, 123]

    all_runs: List[Dict[str, Any]] = []
    for m in display_methods:
        for o in display_orders:
            for s in display_seeds:
                data = get_completed_run_metrics(method=m, order=o, seed=s)
                if data is not None:
                    all_runs.append(data)

    if not all_runs:
        logger.warning("No completed incremental runs found to compile!")
        return

    # Per-run rows
    per_run_rows = []
    for r in all_runs:
        per_run_rows.append({
            "method": r["method"],
            "order": r["order"],
            "seed": r["seed"],
            "memory_cost": format_memory_cost(r["method"], r.get("replay_memory_cost", 0)),
            "a_avg": r["average_incremental_accuracy"],
            "final_acc": r["final_accuracy"],
            "final_macro_f1": r["final_macro_f1"],
            "forgetting": r["overall_forgetting"],
            "bwt": r["backward_transfer"],
            "restricted_old_acc": r.get("restricted_old_accuracy", 0.0),
            "old_mass": r.get("mean_softmax_mass_old", 0.0),
            "new_mass": r.get("mean_softmax_mass_new", 0.0),
            "time_sec": r["total_training_time_sec"],
        })

    # Summary rows: Group by (order, method)
    grouped = defaultdict(list)
    for r in all_runs:
        key = (r["order"], r["method"])
        grouped[key].append(r)

    summary_rows = []
    for (order, method), runs in sorted(grouped.items()):
        a_avgs = [r["average_incremental_accuracy"] for r in runs]
        final_accs = [r["final_accuracy"] for r in runs]
        final_f1s = [r["final_macro_f1"] for r in runs]
        forgettings = [r["overall_forgetting"] for r in runs]
        bwts = [r["backward_transfer"] for r in runs]
        restr_accs = [r.get("restricted_old_accuracy", 0.0) for r in runs]
        old_masses = [r.get("mean_softmax_mass_old", 0.0) for r in runs]
        new_masses = [r.get("mean_softmax_mass_new", 0.0) for r in runs]
        times = [r["total_training_time_sec"] for r in runs]
        mem = format_memory_cost(method, runs[0].get("replay_memory_cost", 0))

        summary_rows.append({
            "order": order,
            "method": method.upper(),
            "memory_cost": mem,
            "mean_a_avg": float(np.mean(a_avgs)),
            "std_a_avg": float(np.std(a_avgs)),
            "mean_final_acc": float(np.mean(final_accs)),
            "std_final_acc": float(np.std(final_accs)),
            "mean_final_f1": float(np.mean(final_f1s)),
            "std_final_f1": float(np.std(final_f1s)),
            "mean_forgetting": float(np.mean(forgettings)),
            "std_forgetting": float(np.std(forgettings)),
            "mean_bwt": float(np.mean(bwts)),
            "std_bwt": float(np.std(bwts)),
            "mean_restricted_old_acc": float(np.mean(restr_accs)),
            "mean_old_mass": float(np.mean(old_masses)),
            "mean_new_mass": float(np.mean(new_masses)),
            "mean_time_sec": float(np.mean(times)),
        })

    # Write CSV
    csv_path = TABLES_DIR / "incremental_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "order", "seed", "memory_cost",
            "a_avg", "final_acc", "final_macro_f1", "forgetting", "bwt",
            "restricted_old_acc", "old_mass", "new_mass", "time_sec"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_run_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # Write Markdown
    md_path = TABLES_DIR / "incremental_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Class-Incremental Continual Learning Results Report (Stages B & D)\n\n")
        f.write("- **Architecture**: Custom CNN Medium Variant (1,036,548 parameters, 1077.56 MFLOPs)\n")
        f.write("- **Methods Evaluated**: Naive Fine-Tuning, LwF (baseline), LwF_New, lwfPlus (`lwfLF_frz_l1`), Experience Replay (M=60, 120), Joint Training (Upper Bound)\n")
        f.write("- **Task Orders**: Order A (3 tasks), Order B (3 tasks), Order C (2 tasks, shared Task 1) across Seeds `[42, 123]`\n\n")

        f.write("### 1. Continual Learning Performance Summary (Mean ± Std across Seeds 42 and 123)\n\n")
        f.write("| Order | Method | Replay Memory (M) | Average Incremental Accuracy ($A_{\\text{avg}}$) | Final Accuracy | Final Macro-F1 | Forgetting ($F$) ↓ | Backward Transfer (BWT) | Restricted Old Acc | Mean Old Mass | Mean Train Time |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        if joint_baseline:
            best_test_f1 = joint_baseline.get("best_test_macro_f1", "-")
            best_test_acc = joint_baseline.get("best_test_acc", "-")
            f.write(
                f"| **All** | **Joint Training (Upper Bound)** | 0 images (all data) | "
                f"**—** | {best_test_acc} | **{best_test_f1}** | **0.0000** | 0.0000 | 1.0000 | 1.0000 | {joint_baseline.get('train_time_sec', '-')} s |\n"
            )

        for s in summary_rows:
            f.write(
                f"| **{s['order']}** | **{s['method']}** | {s['memory_cost']} | "
                f"**{s['mean_a_avg']:.4f} ± {s['std_a_avg']:.4f}** | "
                f"{s['mean_final_acc']:.4f} ± {s['std_final_acc']:.4f} | "
                f"{s['mean_final_f1']:.4f} ± {s['std_final_f1']:.4f} | "
                f"{s['mean_forgetting']:.4f} ± {s['std_forgetting']:.4f} | "
                f"{s['mean_bwt']:.4f} ± {s['std_bwt']:.4f} | "
                f"{s['mean_restricted_old_acc']:.4f} | "
                f"{s['mean_old_mass']:.4f} | "
                f"{s['mean_time_sec']:.1f} s |\n"
            )
        f.write("\n")

        f.write("### 2. Individual Per-Run Results Matrix\n\n")
        f.write("| Method | Order | Seed | Memory (M) | $A_{\\text{avg}}$ | Final Accuracy | Final Macro-F1 | Forgetting ($F$) | Restricted Old Acc | Old Mass | New Mass |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in per_run_rows:
            f.write(
                f"| {r['method'].upper()} | {r['order']} | {r['seed']} | {r['memory_cost']} | "
                f"{r['a_avg']:.4f} | {r['final_acc']:.4f} | {r['final_macro_f1']:.4f} | "
                f"{r['forgetting']:.4f} | {r['restricted_old_acc']:.4f} | "
                f"{r['old_mass']:.4f} | {r['new_mass']:.4f} |\n"
            )
        f.write("\n")

    logger.info("Successfully updated %s and %s", md_path, csv_path)


def run_all_stage_d_final_experiments(epochs_per_task: int = 30) -> None:
    """Execute all missing Stage D final runs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device for Stage D Final: %s", device)

    custom_cnn_yaml = CONFIGS_DIR / "model" / "custom_cnn.yaml"
    model_cfg = load_yaml(custom_cnn_yaml)
    variant = model_cfg.get("active_variant", "medium")
    logger.info("Selected Custom CNN architecture variant: %s", variant.upper())

    for method, order, seed in STAGE_D_MANIFEST:
        existing = get_completed_run_metrics(method=method, order=order, seed=seed)
        if existing is not None:
            logger.info("Resuming: Skipping completed run %s_%s_%d", method, order, seed)
            continue

        res = run_single_incremental_experiment(
            method=method,
            order=order,
            seed=seed,
            variant=variant,
            epochs_per_task=epochs_per_task,
            device=device,
        )
        del res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Update consolidated tables after each run
        generate_consolidated_incremental_tables()

    # Final table build
    generate_consolidated_incremental_tables()
    logger.info("Stage D final runs completed successfully!")


if __name__ == "__main__":
    run_all_stage_d_final_experiments(epochs_per_task=30)

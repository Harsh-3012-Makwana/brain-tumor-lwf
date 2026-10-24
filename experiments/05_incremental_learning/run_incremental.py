"""Execution script for all 16 class-incremental learning runs on the Custom CNN.

Runs 4 methods (finetune, lwf, replay60, replay120) x 2 orders (A, B) x 2 seeds (42, 123).
Supports resumability, smoke-testing, sanity-checking, and generates incremental_results.md/.csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Insert project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.incremental.runner.incremental_runner import run_single_incremental_experiment
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, RUNS_DIR, TABLES_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("incremental_experiment")

METHODS = ["finetune", "lwf", "replay60", "replay120"]
ORDERS = ["A", "B"]
SEEDS = [42, 123]


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


def run_smoke_test(variant: str, device: torch.device) -> None:
    """Run 1 epoch per task for all 4 methods on Order A, Seed 42 to verify the pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING SMOKE TEST (1 epoch/task on Order A, Seed 42 for all 4 methods)")
    logger.info("=" * 70)

    smoke_runs_dir = RUNS_DIR / "_smoke_test"
    smoke_runs_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        logger.info("--- Testing method: %s ---", method.upper())
        res = run_single_incremental_experiment(
            method=method,
            order="A",
            seed=42,
            variant=variant,
            epochs_per_task=1,
            device=device,
            base_runs_dir=smoke_runs_dir,
        )
        assert "average_incremental_accuracy" in res
        assert "overall_forgetting" in res
        assert len(res["accuracy_matrix"]) == 3
        logger.info("Smoke test for %s succeeded! A_avg = %.4f", method, res["average_incremental_accuracy"])

    # Clean up smoke test run folder
    logger.info("Cleaning up smoke test directory: %s", smoke_runs_dir)
    shutil.rmtree(smoke_runs_dir)
    logger.info("Smoke test passed completely!")


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


def generate_incremental_tables(all_results: List[Dict[str, Any]]) -> None:
    """Compile publication-quality Markdown and CSV incremental results tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    joint_baseline = load_joint_baseline_metrics()

    # 1. Per-run rows
    per_run_rows: List[Dict[str, Any]] = []
    for r in all_results:
        per_run_rows.append({
            "method": r["method"],
            "order": r["order"],
            "seed": r["seed"],
            "memory_cost": r["replay_memory_cost"],
            "a_avg": r["average_incremental_accuracy"],
            "final_acc": r["final_accuracy"],
            "final_macro_f1": r["final_macro_f1"],
            "forgetting": r["overall_forgetting"],
            "bwt": r["backward_transfer"],
            "r_matrix": str(r["accuracy_matrix"]),
            "time_sec": r["total_training_time_sec"],
        })

    # 2. Grouped Summary rows: Group by (order, method)
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["order"], r["method"])
        grouped[key].append(r)

    summary_rows: List[Dict[str, Any]] = []
    for (order, method), runs in sorted(grouped.items()):
        a_avgs = [r["average_incremental_accuracy"] for r in runs]
        final_accs = [r["final_accuracy"] for r in runs]
        final_f1s = [r["final_macro_f1"] for r in runs]
        forgettings = [r["overall_forgetting"] for r in runs]
        bwts = [r["backward_transfer"] for r in runs]
        times = [r["total_training_time_sec"] for r in runs]
        mem = runs[0]["replay_memory_cost"]

        summary_rows.append({
            "order": order,
            "method": method,
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
            "mean_time_sec": float(np.mean(times)),
        })

    # Write CSV
    csv_path = TABLES_DIR / "incremental_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "order", "seed", "memory_cost",
            "a_avg", "final_acc", "final_macro_f1", "forgetting", "bwt", "time_sec"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_run_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # Write Markdown Table
    md_path = TABLES_DIR / "incremental_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Class-Incremental Continual Learning Results Report\n\n")
        f.write("- **Architecture**: Custom CNN (<5M parameters, active variant)\n")
        f.write("- **Methods Evaluated**: Naive Fine-Tuning (lower bound), LwF (distillation), Experience Replay (M=60, 120), Joint Training (upper bound oracle)\n")
        f.write("- **Task Orders**: Order A and Order B across random seeds `[42, 123]`\n\n")

        f.write("### 1. Continual Learning Performance Summary (Mean ± Std across Seeds 42 and 123)\n\n")
        f.write("| Order | Method | Replay Memory (M) | Average Incremental Accuracy ($A_{\\text{avg}}$) | Final Accuracy | Final Macro-F1 | Forgetting ($F$) ↓ | Backward Transfer (BWT) | Mean Train Time |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        # Joint upper bound oracle row if available
        if joint_baseline:
            best_val_f1 = joint_baseline.get("best_val_macro_f1", "-")
            best_test_f1 = joint_baseline.get("best_test_macro_f1", "-")
            best_test_acc = joint_baseline.get("best_test_acc", "-")
            f.write(
                f"| **All** | **Joint Training (Upper Bound)** | 0 images | "
                f"**—** | {best_test_acc} | **{best_test_f1}** | **0.0000** | 0.0000 | {joint_baseline.get('train_time_sec', '-')} s |\n"
            )

        for s in summary_rows:
            f.write(
                f"| **{s['order']}** | **{s['method'].upper()}** | {s['memory_cost']} images | "
                f"**{s['mean_a_avg']:.4f} ± {s['std_a_avg']:.4f}** | "
                f"{s['mean_final_acc']:.4f} ± {s['std_final_acc']:.4f} | "
                f"{s['mean_final_f1']:.4f} ± {s['std_final_f1']:.4f} | "
                f"{s['mean_forgetting']:.4f} ± {s['std_forgetting']:.4f} | "
                f"{s['mean_bwt']:.4f} ± {s['std_bwt']:.4f} | "
                f"{s['mean_time_sec']:.1f} s |\n"
            )
        f.write("\n")

        f.write("### 2. Individual Per-Run Results Matrix\n\n")
        f.write("| Method | Order | Seed | Memory (M) | $A_{\\text{avg}}$ | Final Accuracy | Final Macro-F1 | Forgetting ($F$) | Backward Transfer (BWT) | Accuracy Matrix $R$ |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|\n")
        for r in per_run_rows:
            f.write(
                f"| {r['method'].upper()} | {r['order']} | {r['seed']} | {r['memory_cost']} | "
                f"{r['a_avg']:.4f} | {r['final_acc']:.4f} | {r['final_macro_f1']:.4f} | "
                f"{r['forgetting']:.4f} | {r['bwt']:.4f} | `{r['r_matrix']}` |\n"
            )
        f.write("\n")


def check_sanity_rules(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify continual learning sanity rules:
    1. Naive Fine-Tuning shows catastrophic forgetting of early classes.
    2. LwF and Replay retain knowledge better than Fine-Tuning (lower F, higher A_avg).
    """
    logger.info("Executing sanity checks on incremental learning results...")

    ft_runs = [r for r in all_results if r["method"] == "finetune"]
    lwf_runs = [r for r in all_results if r["method"] == "lwf"]
    rep60_runs = [r for r in all_results if r["method"] == "replay60"]
    rep120_runs = [r for r in all_results if r["method"] == "replay120"]

    ft_mean_f = float(np.mean([r["overall_forgetting"] for r in ft_runs]))
    lwf_mean_f = float(np.mean([r["overall_forgetting"] for r in lwf_runs]))
    rep60_mean_f = float(np.mean([r["overall_forgetting"] for r in rep60_runs]))
    rep120_mean_f = float(np.mean([r["overall_forgetting"] for r in rep120_runs]))

    ft_mean_a = float(np.mean([r["average_incremental_accuracy"] for r in ft_runs]))
    lwf_mean_a = float(np.mean([r["average_incremental_accuracy"] for r in lwf_runs]))
    rep60_mean_a = float(np.mean([r["average_incremental_accuracy"] for r in rep60_runs]))
    rep120_mean_a = float(np.mean([r["average_incremental_accuracy"] for r in rep120_runs]))

    findings = {
        "ft_catastrophic_forgetting_observed": ft_mean_f > 0.30,
        "ft_mean_forgetting": round(ft_mean_f, 4),
        "lwf_mean_forgetting": round(lwf_mean_f, 4),
        "rep60_mean_forgetting": round(rep60_mean_f, 4),
        "rep120_mean_forgetting": round(rep120_mean_f, 4),
        "ft_mean_a_avg": round(ft_mean_a, 4),
        "lwf_mean_a_avg": round(lwf_mean_a, 4),
        "rep60_mean_a_avg": round(rep60_mean_a, 4),
        "rep120_mean_a_avg": round(rep120_mean_a, 4),
        "lwf_retains_better_than_ft": lwf_mean_f < ft_mean_f,
        "replay_retains_better_than_ft": rep60_mean_f < ft_mean_f and rep120_mean_f < ft_mean_f,
    }

    logger.info("Sanity check findings: %s", findings)
    return findings


def run_all_incremental_experiments(
    epochs_per_task: int = 30,
    skip_smoke_test: bool = False,
    target_methods: Optional[List[str]] = None,
    target_orders: Optional[List[str]] = None,
    target_seeds: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute class-incremental runs sequentially."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device for Incremental Learning: %s", device)

    methods = [m.lower() for m in target_methods] if target_methods else METHODS
    orders = [o.upper() for o in target_orders] if target_orders else ORDERS
    seeds = target_seeds if target_seeds else SEEDS

    # Load active variant
    custom_cnn_yaml = CONFIGS_DIR / "model" / "custom_cnn.yaml"
    model_cfg = load_yaml(custom_cnn_yaml)
    variant = model_cfg.get("active_variant", "medium")
    logger.info("Selected Custom CNN architecture variant: %s", variant.upper())

    # 1. Smoke test
    if not skip_smoke_test:
        run_smoke_test(variant=variant, device=device)

    # 2. Execute target runs sequentially
    for method in methods:
        for order in orders:
            for seed in seeds:
                # Check resumability
                existing = get_completed_run_metrics(method=method, order=order, seed=seed)
                if existing is not None:
                    logger.info("Resuming: Skipping already completed run %s_%s_%d", method, order, seed)
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

                # Incrementally update tables after each run
                completed_so_far = []
                for m_check in METHODS:
                    for o_check in ORDERS:
                        for s_check in SEEDS:
                            d_check = get_completed_run_metrics(method=m_check, order=o_check, seed=s_check)
                            if d_check is not None:
                                completed_so_far.append(d_check)
                if completed_so_far:
                    generate_incremental_tables(completed_so_far)

    # 3. Gather all completed runs across the entire 16-run manifest
    all_completed: List[Dict[str, Any]] = []
    for m in METHODS:
        for o in ORDERS:
            for s in SEEDS:
                data = get_completed_run_metrics(method=m, order=o, seed=s)
                if data is not None:
                    all_completed.append(data)

    # Compile tables
    if all_completed:
        generate_incremental_tables(all_completed)

    # 4. Check sanity rules if all 16 runs are done
    sanity_summary: Dict[str, Any] = {}
    if len(all_completed) == 16:
        sanity_summary = check_sanity_rules(all_completed)

    return all_completed, sanity_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 16 class-incremental continual learning experiments")
    parser.add_argument("--epochs", type=int, default=30, help="Per-task epoch budget E (default: 30)")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke test")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None, help="Specific methods to run")
    parser.add_argument("--orders", nargs="+", choices=ORDERS, default=None, help="Specific orders to run")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=None, help="Specific seeds to run")
    args = parser.parse_args()

    run_all_incremental_experiments(
        epochs_per_task=args.epochs,
        skip_smoke_test=args.skip_smoke,
        target_methods=args.methods,
        target_orders=args.orders,
        target_seeds=args.seeds,
    )


if __name__ == "__main__":
    main()

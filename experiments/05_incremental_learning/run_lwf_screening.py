"""Execution and selection script for Stage D LwF Screening on Seed 42.

Evaluates 6 configs:
  - lwfL_l1, lwfL_l10 (Variant V2: Logit MSE)
  - lwfLF_l1, lwfLF_l10 (Variant V3: Logit MSE + Feature Cosine)
  - lwfLF_frz_l1, lwfLF_frz_l10 (Variant V4: V3 + Freeze Conv Blocks 1 & 2)
Across Order A and Order C on Seed 42 (12 runs total).

Applies the frozen selection rule:
  Highest mean final Macro-F1 across Order A and Order C on the VALIDATION split of seed 42.
  Ties: simpler variant (V2 < V3 < V4), then smaller lambda.
Generates results/tables/lwf_screening.md and lwf_screening.csv.
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
logger = logging.getLogger("lwf_screening")

SCREENING_CONFIGS = [
    "lwfl_l1",
    "lwfl_l10",
    "lwflf_l1",
    "lwflf_l10",
    "lwflf_frz_l1",
    "lwflf_frz_l10",
]
SCREENING_ORDERS = ["A", "C"]
SCREENING_SEED = 42

VARIANT_COMPLEXITY = {
    "lwfl_l1": (2, 1),
    "lwfl_l10": (2, 10),
    "lwflf_l1": (3, 1),
    "lwflf_l10": (3, 10),
    "lwflf_frz_l1": (4, 1),
    "lwflf_frz_l10": (4, 10),
}


def get_completed_screening_metrics(method: str, order: str, seed: int = 42) -> Optional[Dict[str, Any]]:
    """Retrieve metrics if screening run has already completed."""
    pattern = f"{method.lower()}_{order.lower()}_{seed}_"
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir() and p.name.startswith(pattern):
            metrics_file = p / "metrics.json"
            if metrics_file.is_file():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "final_val_macro_f1" in data and "final_macro_f1" in data:
                        return data
                except Exception:
                    pass
    return None


def generate_screening_tables(all_results: List[Dict[str, Any]], selected_config: str) -> None:
    """Generate Markdown and CSV screening tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Group by config
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["method"]].append(r)

    summary_rows = []
    for cfg in SCREENING_CONFIGS:
        runs = grouped.get(cfg, [])
        if not runs:
            continue
        val_f1s = [r["final_val_macro_f1"] for r in runs]
        test_f1s = [r["final_macro_f1"] for r in runs]
        test_accs = [r["final_accuracy"] for r in runs]
        forgettings = [r["overall_forgetting"] for r in runs]
        restr_accs = [r["restricted_old_accuracy"] for r in runs]
        old_masses = [r["mean_softmax_mass_old"] for r in runs]
        new_masses = [r["mean_softmax_mass_new"] for r in runs]

        summary_rows.append({
            "config": cfg,
            "mean_val_macro_f1": float(np.mean(val_f1s)),
            "mean_test_macro_f1": float(np.mean(test_f1s)),
            "mean_test_accuracy": float(np.mean(test_accs)),
            "mean_forgetting": float(np.mean(forgettings)),
            "mean_restricted_old_acc": float(np.mean(restr_accs)),
            "mean_mass_old": float(np.mean(old_masses)),
            "mean_mass_new": float(np.mean(new_masses)),
            "selected": "YES (SELECTED)" if cfg == selected_config else "NO",
        })

    # Write CSV
    csv_path = TABLES_DIR / "lwf_screening.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config", "order", "val_macro_f1", "test_macro_f1", "test_accuracy",
            "forgetting", "restricted_old_accuracy", "mean_softmax_mass_old",
            "mean_softmax_mass_new", "selected"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "config": r["method"],
                "order": r["order"],
                "val_macro_f1": r["final_val_macro_f1"],
                "test_macro_f1": r["final_macro_f1"],
                "test_accuracy": r["final_accuracy"],
                "forgetting": r["overall_forgetting"],
                "restricted_old_accuracy": r["restricted_old_accuracy"],
                "mean_softmax_mass_old": r["mean_softmax_mass_old"],
                "mean_softmax_mass_new": r["mean_softmax_mass_new"],
                "selected": "YES (SELECTED)" if r["method"] == selected_config else "NO",
            })

    # Write Markdown
    md_path = TABLES_DIR / "lwf_screening.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage D: LwF Architectural Screening Results Report\n\n")
        f.write("- **Hardware / Model**: Custom CNN Medium Variant (1.04M params) on CUDA\n")
        f.write("- **Screening Budget**: 30 epochs per task on Seed 42\n")
        f.write("- **Selection Rule**: Highest mean final validation Macro-F1 across Order A and Order C on Seed 42 (includes old-class validation labels). Test set is strictly isolated.\n")
        f.write(f"- **Selected Config**: **`{selected_config}`**\n\n")

        f.write("### 1. Screening Configurations Aggregated Summary (Order A & Order C)\n\n")
        f.write("| Config | Mean Val Macro-F1 (Selection Metric) | Mean Test Macro-F1 | Mean Test Accuracy | Mean Forgetting ($F$) ↓ | Restricted Old Acc | Mean Old Mass | Selected? |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for s in summary_rows:
            f.write(
                f"| `{s['config']}` | **{s['mean_val_macro_f1']:.4f}** | "
                f"{s['mean_test_macro_f1']:.4f} | {s['mean_test_accuracy']:.4f} | "
                f"{s['mean_forgetting']:.4f} | {s['mean_restricted_old_acc']:.4f} | "
                f"{s['mean_mass_old']:.4f} | **{s['selected']}** |\n"
            )
        f.write("\n")

        f.write("### 2. Individual Screening Runs (12 Runs on Seed 42)\n\n")
        f.write("| Config | Order | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Forgetting ($F$) | Restricted Old Acc | Mean Old Mass | Mean New Mass |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in all_results:
            f.write(
                f"| `{r['method']}` | {r['order']} | {r['final_val_macro_f1']:.4f} | "
                f"{r['final_macro_f1']:.4f} | {r['final_accuracy']:.4f} | "
                f"{r['overall_forgetting']:.4f} | {r['restricted_old_accuracy']:.4f} | "
                f"{r['mean_softmax_mass_old']:.4f} | {r['mean_softmax_mass_new']:.4f} |\n"
            )
        f.write("\n")


def apply_selection_rule(all_results: List[Dict[str, Any]]) -> str:
    """Select winning config based on highest mean validation Macro-F1, with frozen tie-breaker."""
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["method"]].append(r)

    scores = {}
    for cfg in SCREENING_CONFIGS:
        runs = grouped.get(cfg, [])
        if len(runs) == 2:
            mean_val_f1 = float(np.mean([r["final_val_macro_f1"] for r in runs]))
            complexity, lam = VARIANT_COMPLEXITY[cfg]
            # Key: (val_f1, -complexity, -lam)
            scores[cfg] = (mean_val_f1, -complexity, -lam)

    if not scores:
        raise RuntimeError("No completed screening runs found!")

    # Sort descending by key
    sorted_cfgs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_cfgs[0][0]
    logger.info("=" * 70)
    logger.info("SCREENING SELECTION OUTCOME: Winner is %s (Score: %s)", winner, sorted_cfgs[0][1])
    logger.info("All ranking: %s", [(c, s[0]) for c, s in sorted_cfgs])
    logger.info("=" * 70)
    return winner


def run_screening_experiments(epochs_per_task: int = 30) -> Tuple[List[Dict[str, Any]], str]:
    """Execute all 12 screening runs sequentially."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device for Stage D Screening: %s", device)

    custom_cnn_yaml = CONFIGS_DIR / "model" / "custom_cnn.yaml"
    model_cfg = load_yaml(custom_cnn_yaml)
    variant = model_cfg.get("active_variant", "medium")
    logger.info("Selected Custom CNN architecture variant: %s", variant.upper())

    all_results: List[Dict[str, Any]] = []

    for cfg in SCREENING_CONFIGS:
        for order in SCREENING_ORDERS:
            existing = get_completed_screening_metrics(method=cfg, order=order, seed=SCREENING_SEED)
            if existing is not None:
                logger.info("Resuming: Skipping completed screening run %s_%s", cfg, order)
                all_results.append(existing)
                continue

            res = run_single_incremental_experiment(
                method=cfg,
                order=order,
                seed=SCREENING_SEED,
                variant=variant,
                epochs_per_task=epochs_per_task,
                device=device,
            )
            all_results.append(res)
            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    winner = apply_selection_rule(all_results)
    generate_screening_tables(all_results, selected_config=winner)
    return all_results, winner


if __name__ == "__main__":
    run_screening_experiments(epochs_per_task=30)

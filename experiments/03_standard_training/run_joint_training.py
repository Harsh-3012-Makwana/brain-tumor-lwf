"""Joint Training pipeline for Custom CNN variants and empirical main-variant selection."""

from __future__ import annotations

import csv
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Insert project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tasks.builder import get_joint_datasets
from src.evaluation.metrics.classification import evaluate_loader
from src.evaluation.plots.confusion_matrix import plot_confusion_matrix
from src.evaluation.plots.training_curves import plot_training_curves
from src.models.custom_cnn.factory import build_model
from src.models.utils.profiling import count_flops, count_parameters
from src.training.trainers.base_trainer import build_optimizer_and_scheduler, fit
from src.training.trainers.run_context import RunContext
from src.utils.config import load_yaml, save_yaml
from src.utils.paths import CONFIGS_DIR, FIGURES_DIR, RUNS_DIR, TABLES_DIR
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("joint_training")

VARIANTS = ["small", "medium", "large"]
SEEDS = [42, 123]
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def seed_worker(worker_id: int) -> None:
    """Deterministic worker initialization function for PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def run_single_joint_experiment(
    variant: str,
    seed: int,
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute a single joint training experiment for a specific variant and seed."""
    logger.info("=" * 65)
    logger.info("STARTING JOINT RUN: variant=%s, seed=%d", variant.upper(), seed)
    logger.info("=" * 65)

    # 1. Deterministic seeding
    set_seed(seed, deterministic=True)

    # 2. Hyperparameters
    batch_size = training_cfg.get("batch_size", 64)
    num_epochs = training_cfg.get("num_epochs", 40)
    lr = training_cfg.get("optimizer", {}).get("lr", 0.001)
    weight_decay = training_cfg.get("optimizer", {}).get("weight_decay", 0.0001)
    warmup_epochs = training_cfg.get("scheduler", {}).get("warmup_epochs", 2)
    min_lr = training_cfg.get("scheduler", {}).get("min_lr", 1e-6)
    patience = training_cfg.get("early_stopping", {}).get("patience", 8)
    min_delta = training_cfg.get("early_stopping", {}).get("min_delta", 0.0005)
    grad_clip = training_cfg.get("gradient_clip_val", 1.0)
    use_amp = training_cfg.get("use_amp", True)
    num_workers = training_cfg.get("num_workers", 0)

    # 3. Datasets and Loaders
    train_ds, val_ds, test_ds = get_joint_datasets(seed=seed, image_size=(150, 150))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(device.type == "cuda"),
    )

    # 4. Initialize Run Context
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ctx = RunContext(
        method="joint",
        order="none",
        seed=seed,
        timestamp=f"{variant}_{timestamp}",
        base_runs_dir=RUNS_DIR,
    )
    run_ctx.create_tensorboard_writer()
    run_ctx.save_env_info()

    # Save active snapshot config
    snapshot_cfg = {
        "variant": variant,
        "seed": seed,
        "training": training_cfg,
    }
    run_ctx.save_config_snapshot(snapshot_cfg)

    # 5. Build Model
    model = build_model(variant=variant, num_classes=4).to(device)
    params_info = count_parameters(model)
    flops = count_flops(model, input_size=(1, 3, 150, 150), device="cpu")
    model = model.to(device)

    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 6. Optimizer and Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
    )

    loss_fn = nn.CrossEntropyLoss()

    # 7. Training
    t_start = time.time()
    best_model, best_val_summary, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        gradient_clip_val=grad_clip,
        use_amp=use_amp,
        run_context=run_ctx,
        class_names=CLASS_NAMES,
        verbose=True,
    )
    training_time_sec = time.time() - t_start

    peak_gpu_mem_mb = 0.0
    if device.type == "cuda":
        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    # 8. Post-training Evaluation on Val and Official Test Set
    val_eval = evaluate_loader(
        best_model, val_loader, device=device, criterion=loss_fn, class_names=CLASS_NAMES, num_classes=4
    )
    test_eval = evaluate_loader(
        best_model, test_loader, device=device, criterion=loss_fn, class_names=CLASS_NAMES, num_classes=4
    )

    # 9. Save metrics and plots
    metrics_summary = {
        "variant": variant,
        "seed": seed,
        "total_params": params_info["total"],
        "flops": flops,
        "training_time_sec": round(training_time_sec, 2),
        "peak_gpu_mem_mb": round(peak_gpu_mem_mb, 2),
        "best_epoch": best_val_summary.get("epoch", 0),
        "val_loss": round(val_eval["loss"], 5),
        "val_accuracy": round(val_eval["accuracy"], 4),
        "val_macro_f1": round(val_eval["macro_f1"], 4),
        "val_per_class": val_eval["per_class"],
        "test_loss": round(test_eval["loss"], 5),
        "test_accuracy": round(test_eval["accuracy"], 4),
        "test_macro_f1": round(test_eval["macro_f1"], 4),
        "test_per_class": test_eval["per_class"],
        "test_confusion_matrix": test_eval["confusion_matrix"],
    }
    run_ctx.save_metrics(metrics_summary)

    # Plot training curves
    curve_fig_path = FIGURES_DIR / f"joint_curves_{variant}_seed{seed}.png"
    plot_training_curves(
        history,
        output_path=curve_fig_path,
        title=f"Joint Training Curves: Custom CNN ({variant.capitalize()}) - Seed {seed}",
    )

    run_ctx.close()

    logger.info(
        "FINISHED JOINT RUN: %s (Seed %d) in %.1fs (Peak GPU Mem: %.1f MB) | Val F1: %.4f, Test F1: %.4f",
        variant.upper(),
        seed,
        training_time_sec,
        peak_gpu_mem_mb,
        val_eval["macro_f1"],
        test_eval["macro_f1"],
    )

    metrics_summary["history"] = history
    return metrics_summary


def run_all_joint_experiments() -> Tuple[str, Dict[str, Any]]:
    """Execute all 6 joint training runs (3 variants x 2 seeds) and select the main variant."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device for Joint Training: %s", device)

    # Load default training config
    training_cfg_path = CONFIGS_DIR / "training" / "default.yaml"
    training_cfg = load_yaml(training_cfg_path)

    all_results: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        for seed in SEEDS:
            res = run_single_joint_experiment(
                variant=variant,
                seed=seed,
                training_cfg=training_cfg,
                device=device,
            )
            all_results.append(res)

    # ------------------ Aggregate Results Per Variant ------------------
    variant_stats: Dict[str, Dict[str, Any]] = {}
    grouped_by_variant = defaultdict(list)
    for r in all_results:
        grouped_by_variant[r["variant"]].append(r)

    comparison_rows: List[Dict[str, Any]] = []

    for var in VARIANTS:
        runs = grouped_by_variant[var]
        val_f1s = [r["val_macro_f1"] for r in runs]
        val_accs = [r["val_accuracy"] for r in runs]
        test_f1s = [r["test_macro_f1"] for r in runs]
        test_accs = [r["test_accuracy"] for r in runs]
        times = [r["training_time_sec"] for r in runs]
        mems = [r["peak_gpu_mem_mb"] for r in runs]

        mean_val_f1 = float(np.mean(val_f1s))
        std_val_f1 = float(np.std(val_f1s))
        mean_val_acc = float(np.mean(val_accs))
        std_val_acc = float(np.std(val_accs))
        mean_test_f1 = float(np.mean(test_f1s))
        std_test_f1 = float(np.std(test_f1s))
        mean_test_acc = float(np.mean(test_accs))
        std_test_acc = float(np.std(test_accs))
        mean_time = float(np.mean(times))
        peak_mem = float(np.max(mems))
        params = runs[0]["total_params"]
        mflops = round(runs[0]["flops"] / 1e6, 2)

        stats = {
            "variant": var,
            "params": params,
            "mflops": mflops,
            "mean_val_macro_f1": round(mean_val_f1, 4),
            "std_val_macro_f1": round(std_val_f1, 4),
            "mean_val_acc": round(mean_val_acc, 4),
            "std_val_acc": round(std_val_acc, 4),
            "mean_test_macro_f1": round(mean_test_f1, 4),
            "std_test_macro_f1": round(std_test_f1, 4),
            "mean_test_acc": round(mean_test_acc, 4),
            "std_test_acc": round(std_test_acc, 4),
            "mean_training_time_sec": round(mean_time, 1),
            "peak_gpu_mem_mb": round(peak_mem, 1),
        }
        variant_stats[var] = stats
        comparison_rows.append(stats)

    # ------------------ Apply Main Variant Selection Rule ------------------
    # Rule: Choose the SMALLEST variant whose mean validation Macro-F1 across seeds [42, 123]
    # is within 1 percentage point (0.01) of the best variant. Selection uses validation only.
    best_variant_f1 = max(s["mean_val_macro_f1"] for s in variant_stats.values())
    threshold_f1 = best_variant_f1 - 0.01

    candidates = [
        var for var in VARIANTS
        if variant_stats[var]["mean_val_macro_f1"] >= threshold_f1
    ]
    # Pick smallest variant among candidates (order in VARIANTS is small < medium < large)
    chosen_variant = candidates[0]

    logger.info("=" * 65)
    logger.info("MAIN VARIANT SELECTION RESULTS:")
    logger.info("Best Val Macro-F1: %.4f (Threshold within 1 pp: %.4f)", best_variant_f1, threshold_f1)
    logger.info("Eligible Candidates: %s", candidates)
    logger.info("Selected Main Variant: %s", chosen_variant.upper())
    logger.info("=" * 65)

    # ------------------ Save Comparison Tables ------------------
    # CSV
    csv_path = TABLES_DIR / "joint_variant_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(comparison_rows[0].keys()) + ["selected"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in comparison_rows:
            r_copy = dict(r)
            r_copy["selected"] = "YES" if r["variant"] == chosen_variant else "NO"
            writer.writerow(r_copy)

    # Markdown
    md_path = TABLES_DIR / "joint_variant_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Joint Training Variant Comparison & Main-Variant Selection Report\n\n")
        f.write(f"- **Evaluated Setup**: Joint Training on all 4 classes across Seeds `[42, 123]`\n")
        f.write(f"- **Selection Rule**: Smallest variant with mean val Macro-F1 within 1 percentage point of best variant\n")
        f.write(f"- **Selected Main Architecture**: **`{chosen_variant.upper()}`**\n\n")

        f.write("| Variant | Params | FLOPs (M) | Val Macro-F1 (Mean ± Std) | Val Accuracy | Test Macro-F1 | Test Accuracy | Mean Train Time | Peak GPU Mem | Selected Main? |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in comparison_rows:
            sel_str = "**YES (MAIN)**" if r["variant"] == chosen_variant else "No"
            f.write(
                f"| **{r['variant'].capitalize()}** | {r['params']:,} | {r['mflops']} M | "
                f"**{r['mean_val_macro_f1']:.4f} ± {r['std_val_macro_f1']:.4f}** | "
                f"{r['mean_val_acc']:.4f} ± {r['std_val_acc']:.4f} | "
                f"{r['mean_test_macro_f1']:.4f} ± {r['std_test_macro_f1']:.4f} | "
                f"{r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f} | "
                f"{r['mean_training_time_sec']:.1f} s | {r['peak_gpu_mem_mb']:.1f} MB | {sel_str} |\n"
            )
        f.write("\n")
        f.write("### Scientific Notes on Validation vs. Test Metrics\n")
        f.write("- All selection and early stopping decisions were made strictly on validation partitions.\n")
        f.write("- Official test set metrics are reported for baseline upper-bound reference only.\n")
        f.write("- Consistent with the Phase 1 audit finding (2,022 near-duplicate train/test pairs), test accuracy scores reflect slice similarity across partitions.\n")

    # ------------------ Plot Confusion Matrix for Chosen Variant (Best Seed) ------------------
    chosen_runs = grouped_by_variant[chosen_variant]
    best_chosen_run = max(chosen_runs, key=lambda x: x["val_macro_f1"])
    cm = best_chosen_run["test_confusion_matrix"]
    cm_path = FIGURES_DIR / f"confusion_matrix_joint_{chosen_variant}.png"
    plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        output_path=cm_path,
        title=f"Joint Training Confusion Matrix: {chosen_variant.capitalize()} (Seed {best_chosen_run['seed']})",
        normalize=True,
    )
    logger.info("Saved chosen variant confusion matrix to %s", cm_path)

    # ------------------ Update active_variant in custom_cnn.yaml ------------------
    custom_cnn_yaml = CONFIGS_DIR / "model" / "custom_cnn.yaml"
    model_cfg = load_yaml(custom_cnn_yaml)
    model_cfg["active_variant"] = chosen_variant
    save_yaml(model_cfg, custom_cnn_yaml)
    logger.info("Updated configs/model/custom_cnn.yaml: active_variant='%s'", chosen_variant)

    return chosen_variant, variant_stats


def main() -> None:
    chosen_var, stats = run_all_joint_experiments()
    logger.info("Phase 3 Joint Training pipeline completed! Selected variant: %s", chosen_var)


if __name__ == "__main__":
    main()

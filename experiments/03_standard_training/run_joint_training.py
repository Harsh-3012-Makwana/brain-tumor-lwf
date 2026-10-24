"""Joint Training pipeline for Custom CNN variants and empirical main-variant selection.

Fixed schedule: 60 epochs, cosine decay to ~0, no early stopping, save best.pt and last.pt.
Reports both best-val and last-epoch metrics on validation and test sets.
Computes diagnostic on aug vs non-aug test error rate and determines per-task epoch budget E.
"""

from __future__ import annotations

import csv
import logging
import shutil
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
from src.evaluation.metrics.classification import compute_classification_metrics, evaluate_loader
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


def move_old_phase3_runs() -> None:
    """Move incomplete or non-30-epoch joint runs to results/runs/_superseded/."""
    superseded_dir = RUNS_DIR / "_superseded"
    superseded_dir.mkdir(parents=True, exist_ok=True)
    import json

    for p in list(RUNS_DIR.iterdir()):
        if p.is_dir() and p.name != "_superseded":
            metrics_file = p / "metrics.json"
            is_valid_30_epoch_run = False
            if metrics_file.is_file():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("num_epochs_total") == 30 and "val_macro_f1" in data:
                        is_valid_30_epoch_run = True
                except Exception:
                    pass
            if not is_valid_30_epoch_run:
                dest = superseded_dir / p.name
                if dest.exists():
                    shutil.rmtree(dest)
                logger.info("Moving incomplete or non-30-epoch run %s to %s", p.name, superseded_dir)
                shutil.move(str(p), str(dest))


def find_existing_30_epoch_run(variant: str, seed: int) -> Optional[Dict[str, Any]]:
    """Check if a 30-epoch joint run for this variant and seed is already completed."""
    import json

    pattern = f"joint_none_{seed}_{variant}_"
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir() and p.name.startswith(pattern):
            metrics_file = p / "metrics.json"
            if metrics_file.is_file():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("num_epochs_total") == 30 and "val_macro_f1" in data:
                        return data
                except Exception:
                    pass
    return None


def evaluate_aug_diagnostic(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Compute test error rate on files whose name contains 'aug' vs files that do not."""
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)
            paths = batch[2] if len(batch) > 2 else []

            logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_paths.extend(paths)

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)

    # Identify aug vs non-aug
    is_aug = np.array(["aug" in Path(p).name.lower() for p in all_paths])

    # Aug subset
    aug_count = int(np.sum(is_aug))
    aug_correct = int(np.sum(targets_arr[is_aug] == preds_arr[is_aug])) if aug_count > 0 else 0
    aug_error_rate = 1.0 - (aug_correct / aug_count) if aug_count > 0 else 0.0

    # Non-aug subset
    non_aug_count = int(np.sum(~is_aug))
    non_aug_correct = int(np.sum(targets_arr[~is_aug] == preds_arr[~is_aug])) if non_aug_count > 0 else 0
    non_aug_error_rate = 1.0 - (non_aug_correct / non_aug_count) if non_aug_count > 0 else 0.0

    # Per-class recall
    overall_metrics = compute_classification_metrics(all_targets, all_preds, class_names=CLASS_NAMES, num_classes=4)
    per_class_recall = {cls_name: overall_metrics["per_class"][cls_name]["recall"] for cls_name in CLASS_NAMES}

    return {
        "aug_count": aug_count,
        "aug_error_rate": round(aug_error_rate, 4),
        "aug_accuracy": round(1.0 - aug_error_rate, 4),
        "non_aug_count": non_aug_count,
        "non_aug_error_rate": round(non_aug_error_rate, 4),
        "non_aug_accuracy": round(1.0 - non_aug_error_rate, 4),
        "per_class_recall": per_class_recall,
    }


def compute_epoch_budget_E(history: List[Dict[str, Any]]) -> Tuple[int, float, int]:
    """Smooth val Macro-F1 with a 3-epoch moving average and find budget E.

    Returns:
        Tuple of (E, max_smoothed_f1, first_epoch_within_0_5_pp)
    """
    f1s = [h["val_macro_f1"] for h in history]
    n = len(f1s)
    smoothed = []
    for i in range(n):
        window = f1s[max(0, i - 1) : min(n, i + 2)]
        smoothed.append(float(np.mean(window)))

    max_smoothed = float(np.max(smoothed))
    threshold = max_smoothed - 0.005  # within 0.5 percentage points

    first_epoch = 1
    for i, s in enumerate(smoothed, start=1):
        if s >= threshold:
            first_epoch = i
            break

    E = int(np.ceil(first_epoch / 5.0) * 5)
    return E, round(max_smoothed, 4), first_epoch


def run_single_joint_experiment(
    variant: str,
    seed: int,
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute a single joint training experiment for a specific variant and seed."""
    logger.info("=" * 65)
    logger.info("STARTING JOINT RUN: variant=%s, seed=%d (30 epochs fixed, no ES)", variant.upper(), seed)
    logger.info("=" * 65)

    # 1. Deterministic seeding
    set_seed(seed, deterministic=True)

    # 2. Hyperparameters
    batch_size = training_cfg.get("batch_size", 64)
    num_epochs = 30  # Fixed 30 epochs as explicitly requested by user
    lr = training_cfg.get("optimizer", {}).get("lr", 0.001)
    weight_decay = training_cfg.get("optimizer", {}).get("weight_decay", 0.0001)
    warmup_epochs = training_cfg.get("scheduler", {}).get("warmup_epochs", 2)
    min_lr = training_cfg.get("scheduler", {}).get("min_lr", 1e-6)
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
        "training": {
            **training_cfg,
            "num_epochs": num_epochs,
            "early_stopping": {"enabled": False},
        },
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

    # 6. Optimizer and Scheduler (cosine decay to ~0 over exactly 60 epochs)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
    )

    loss_fn = nn.CrossEntropyLoss()

    # 7. Training for 60 full epochs, NO early stopping, saves best.pt and last.pt
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
        early_stopping=False,
        restore_best=True,
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

    # 8. Post-training Evaluation on Val and Official Test Set for BOTH best.pt and last.pt
    # Evaluation 8a: Best Checkpoint (restored in best_model)
    val_eval_best = evaluate_loader(
        best_model, val_loader, device=device, criterion=loss_fn, class_names=CLASS_NAMES, num_classes=4
    )
    test_eval_best = evaluate_loader(
        best_model, test_loader, device=device, criterion=loss_fn, class_names=CLASS_NAMES, num_classes=4
    )

    # Evaluation 8b: Last Checkpoint
    last_ckpt_path = run_ctx.run_dir / "last.pt"
    last_model = build_model(variant=variant, num_classes=4).to(device)
    last_ckpt = torch.load(last_ckpt_path, map_location=device)
    last_model.load_state_dict(last_ckpt["model_state_dict"])

    val_eval_last = evaluate_loader(
        last_model, val_loader, device=device, criterion=loss_fn, class_names=CLASS_NAMES, num_classes=4
    )
    test_eval_last = evaluate_loader(
        last_model, test_loader, device=device, criterion=loss_fn, class_names=CLASS_NAMES, num_classes=4
    )

    # Evaluation 8c: Diagnostic on aug vs non-aug test images (for best checkpoint)
    aug_diag = evaluate_aug_diagnostic(best_model, test_loader, device=device)

    # Evaluation 8d: Budget E calculation from training history
    budget_E, max_smooth_f1, first_epoch_within_target = compute_epoch_budget_E(history)

    # 9. Save metrics and plots
    metrics_summary = {
        "variant": variant,
        "seed": seed,
        "total_params": params_info["total"],
        "flops": flops,
        "training_time_sec": round(training_time_sec, 2),
        "peak_gpu_mem_mb": round(peak_gpu_mem_mb, 2),
        "num_epochs_total": num_epochs,
        "best_epoch": best_val_summary.get("epoch", 0),
        # Best checkpoint metrics
        "val_loss": round(val_eval_best["loss"], 5),
        "val_accuracy": round(val_eval_best["accuracy"], 4),
        "val_macro_f1": round(val_eval_best["macro_f1"], 4),
        "val_per_class": val_eval_best["per_class"],
        "test_loss": round(test_eval_best["loss"], 5),
        "test_accuracy": round(test_eval_best["accuracy"], 4),
        "test_macro_f1": round(test_eval_best["macro_f1"], 4),
        "test_per_class": test_eval_best["per_class"],
        "test_confusion_matrix": test_eval_best["confusion_matrix"],
        # Last checkpoint metrics
        "last_epoch": num_epochs,
        "last_val_loss": round(val_eval_last["loss"], 5),
        "last_val_accuracy": round(val_eval_last["accuracy"], 4),
        "last_val_macro_f1": round(val_eval_last["macro_f1"], 4),
        "last_val_per_class": val_eval_last["per_class"],
        "last_test_loss": round(test_eval_last["loss"], 5),
        "last_test_accuracy": round(test_eval_last["accuracy"], 4),
        "last_test_macro_f1": round(test_eval_last["macro_f1"], 4),
        "last_test_per_class": test_eval_last["per_class"],
        "last_test_confusion_matrix": test_eval_last["confusion_matrix"],
        # Diagnostics
        "diagnostic_aug": aug_diag,
        "budget_E_calculation": {
            "recommended_E": budget_E,
            "max_smoothed_val_f1": max_smooth_f1,
            "first_epoch_within_0_5_pp": first_epoch_within_target,
        },
    }
    run_ctx.save_metrics(metrics_summary)

    # Plot training curves
    curve_fig_path = FIGURES_DIR / f"joint_curves_{variant}_seed{seed}.png"
    plot_training_curves(
        history,
        output_path=curve_fig_path,
        title=f"Joint Training Curves: Custom CNN ({variant.capitalize()}) - Seed {seed} ({num_epochs} Epochs)",
    )

    run_ctx.close()

    logger.info(
        "FINISHED JOINT RUN: %s (Seed %d) | Best Val F1: %.4f (ep %d), Test F1: %.4f | Last Val F1: %.4f, Test F1: %.4f",
        variant.upper(),
        seed,
        val_eval_best["macro_f1"],
        best_val_summary.get("epoch", 0),
        test_eval_best["macro_f1"],
        val_eval_last["macro_f1"],
        test_eval_last["macro_f1"],
    )

    metrics_summary["history"] = history
    return metrics_summary


def run_all_joint_experiments() -> Tuple[str, Dict[str, Any], int, Dict[int, Dict[str, Any]]]:
    """Execute all 6 joint training runs (3 variants x 2 seeds) and select the main variant."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean previous runs by archiving them to _superseded
    move_old_phase3_runs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Active compute device for Joint Training: %s", device)

    # Load default training config
    training_cfg_path = CONFIGS_DIR / "training" / "default.yaml"
    training_cfg = load_yaml(training_cfg_path)

    all_results: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        for seed in SEEDS:
            existing = find_existing_30_epoch_run(variant=variant, seed=seed)
            if existing is not None:
                logger.info(
                    "Reusing existing completed 30-epoch joint run for %s (seed %d): Val F1=%.4f, Test F1=%.4f",
                    variant.upper(),
                    seed,
                    existing["val_macro_f1"],
                    existing["test_macro_f1"],
                )
                all_results.append(existing)
                continue

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

    # Per-seed table rows
    per_seed_rows: List[Dict[str, Any]] = []
    for r in all_results:
        per_seed_rows.append({
            "variant": r["variant"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "best_val_macro_f1": r["val_macro_f1"],
            "best_val_acc": r["val_accuracy"],
            "best_test_macro_f1": r["test_macro_f1"],
            "best_test_acc": r["test_accuracy"],
            "last_val_macro_f1": r["last_val_macro_f1"],
            "last_val_acc": r["last_val_accuracy"],
            "last_test_macro_f1": r["last_test_macro_f1"],
            "last_test_acc": r["last_test_accuracy"],
            "train_time_sec": r["training_time_sec"],
        })

    # Mean ± std comparison rows
    summary_rows: List[Dict[str, Any]] = []

    for var in VARIANTS:
        runs = grouped_by_variant[var]
        val_f1s_best = [r["val_macro_f1"] for r in runs]
        val_accs_best = [r["val_accuracy"] for r in runs]
        test_f1s_best = [r["test_macro_f1"] for r in runs]
        test_accs_best = [r["test_accuracy"] for r in runs]

        val_f1s_last = [r["last_val_macro_f1"] for r in runs]
        val_accs_last = [r["last_val_accuracy"] for r in runs]
        test_f1s_last = [r["last_test_macro_f1"] for r in runs]
        test_accs_last = [r["last_test_accuracy"] for r in runs]

        times = [r["training_time_sec"] for r in runs]
        mems = [r["peak_gpu_mem_mb"] for r in runs]

        mean_val_f1 = float(np.mean(val_f1s_best))
        std_val_f1 = float(np.std(val_f1s_best))
        mean_val_acc = float(np.mean(val_accs_best))
        std_val_acc = float(np.std(val_accs_best))
        mean_test_f1 = float(np.mean(test_f1s_best))
        std_test_f1 = float(np.std(test_f1s_best))
        mean_test_acc = float(np.mean(test_accs_best))
        std_test_acc = float(np.std(test_accs_best))

        mean_val_f1_last = float(np.mean(val_f1s_last))
        std_val_f1_last = float(np.std(val_f1s_last))
        mean_test_f1_last = float(np.mean(test_f1s_last))
        std_test_f1_last = float(np.std(test_f1s_last))

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
            "mean_val_macro_f1_last": round(mean_val_f1_last, 4),
            "std_val_macro_f1_last": round(std_val_f1_last, 4),
            "mean_test_macro_f1_last": round(mean_test_f1_last, 4),
            "std_test_macro_f1_last": round(std_test_f1_last, 4),
            "mean_training_time_sec": round(mean_time, 1),
            "peak_gpu_mem_mb": round(peak_mem, 1),
        }
        variant_stats[var] = stats
        summary_rows.append(stats)

    # ------------------ Apply Main Variant Selection Rule ------------------
    # Rule: Smallest variant whose mean validation Macro-F1 across seeds [42, 123]
    # is within 1 percentage point (0.01) of the best variant (using best-val checkpoint).
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

    # ------------------ Per-Task Budget E Calculation for Chosen Variant ------------------
    chosen_runs = grouped_by_variant[chosen_variant]
    chosen_e_budgets = []
    chosen_diagnostics: Dict[int, Dict[str, Any]] = {}

    for r in chosen_runs:
        s = r["seed"]
        diag = r["diagnostic_aug"]
        chosen_diagnostics[s] = diag
        b_e = r["budget_E_calculation"]["recommended_E"]
        chosen_e_budgets.append(b_e)

    # Overall E is max or rounded up of the seeds' recommended budget
    final_E = max(chosen_e_budgets)

    logger.info("Per-Task Epoch Budget E for Stage B: %d epochs (seeds individual: %s)", final_E, chosen_e_budgets)

    # ------------------ Save Comparison Tables ------------------
    # CSV
    csv_path = TABLES_DIR / "joint_variant_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant", "seed", "params", "mflops",
            "best_epoch", "best_val_macro_f1", "best_val_acc", "best_test_macro_f1", "best_test_acc",
            "last_val_macro_f1", "last_val_acc", "last_test_macro_f1", "last_test_acc",
            "train_time_sec", "selected",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Per-seed rows
        for r in per_seed_rows:
            var = r["variant"]
            row_dict = {
                "variant": var,
                "seed": r["seed"],
                "params": variant_stats[var]["params"],
                "mflops": variant_stats[var]["mflops"],
                "best_epoch": r["best_epoch"],
                "best_val_macro_f1": r["best_val_macro_f1"],
                "best_val_acc": r["best_val_acc"],
                "best_test_macro_f1": r["best_test_macro_f1"],
                "best_test_acc": r["best_test_acc"],
                "last_val_macro_f1": r["last_val_macro_f1"],
                "last_val_acc": r["last_val_acc"],
                "last_test_macro_f1": r["last_test_macro_f1"],
                "last_test_acc": r["last_test_acc"],
                "train_time_sec": r["train_time_sec"],
                "selected": "YES" if var == chosen_variant else "NO",
            }
            writer.writerow(row_dict)

        # Summary rows
        for s in summary_rows:
            var = s["variant"]
            row_dict = {
                "variant": f"{var}_MEAN_STD",
                "seed": "all",
                "params": s["params"],
                "mflops": s["mflops"],
                "best_epoch": "mean",
                "best_val_macro_f1": f"{s['mean_val_macro_f1']}±{s['std_val_macro_f1']}",
                "best_val_acc": f"{s['mean_val_acc']}±{s['std_val_acc']}",
                "best_test_macro_f1": f"{s['mean_test_macro_f1']}±{s['std_test_macro_f1']}",
                "best_test_acc": f"{s['mean_test_acc']}±{s['std_test_acc']}",
                "last_val_macro_f1": f"{s['mean_val_macro_f1_last']}±{s['std_val_macro_f1_last']}",
                "last_val_acc": "-",
                "last_test_macro_f1": f"{s['mean_test_macro_f1_last']}±{s['std_test_macro_f1_last']}",
                "last_test_acc": "-",
                "train_time_sec": s["mean_training_time_sec"],
                "selected": "YES (MAIN)" if var == chosen_variant else "NO",
            }
            writer.writerow(row_dict)

    # Markdown Table
    md_path = TABLES_DIR / "joint_variant_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Joint Training Variant Comparison & Main-Variant Selection Report\n\n")
        f.write(f"- **Evaluated Setup**: Fixed 30 epochs, Cosine decay to min_lr=1e-6, NO early stopping across Seeds `[42, 123]`\n")
        f.write(f"- **Selection Rule**: Smallest variant with mean val Macro-F1 (best checkpoint) within 1 percentage point of best variant\n")
        f.write(f"- **Selected Main Architecture**: **`{chosen_variant.upper()}`**\n")
        f.write(f"- **Recommended Per-Task Epoch Budget E**: **{final_E} epochs**\n\n")

        f.write("### 1. Summary Comparison (Mean ± Std across Seeds 42 and 123)\n\n")
        f.write("| Variant | Params | FLOPs (M) | Val Macro-F1 (Best) | Val Macro-F1 (Last) | Test Macro-F1 (Best) | Test Macro-F1 (Last) | Mean Train Time | Peak GPU Mem | Selected Main? |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for s in summary_rows:
            sel_str = "**YES (MAIN)**" if s["variant"] == chosen_variant else "No"
            f.write(
                f"| **{s['variant'].capitalize()}** | {s['params']:,} | {s['mflops']} M | "
                f"**{s['mean_val_macro_f1']:.4f} ± {s['std_val_macro_f1']:.4f}** | "
                f"{s['mean_val_macro_f1_last']:.4f} ± {s['std_val_macro_f1_last']:.4f} | "
                f"{s['mean_test_macro_f1']:.4f} ± {s['std_test_macro_f1']:.4f} | "
                f"{s['mean_test_macro_f1_last']:.4f} ± {s['std_test_macro_f1_last']:.4f} | "
                f"{s['mean_training_time_sec']:.1f} s | {s['peak_gpu_mem_mb']:.1f} MB | {sel_str} |\n"
            )
        f.write("\n")

        f.write("### 2. Per-Seed Individual Run Results\n\n")
        f.write("| Variant | Seed | Best Epoch | Val Macro-F1 (Best) | Val Acc (Best) | Test Macro-F1 (Best) | Test Acc (Best) | Val Macro-F1 (Last) | Test Macro-F1 (Last) | Train Time |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in per_seed_rows:
            f.write(
                f"| {r['variant'].capitalize()} | {r['seed']} | Ep {r['best_epoch']} | "
                f"{r['best_val_macro_f1']:.4f} | {r['best_val_acc']:.4f} | "
                f"{r['best_test_macro_f1']:.4f} | {r['best_test_acc']:.4f} | "
                f"{r['last_val_macro_f1']:.4f} | {r['last_test_macro_f1']:.4f} | "
                f"{r['train_time_sec']:.1f} s |\n"
            )
        f.write("\n")

        f.write("### 3. Diagnostic Analysis: 'aug' vs. Non-'aug' Test Slices for Chosen Variant\n\n")
        for s in SEEDS:
            diag = chosen_diagnostics[s]
            f.write(f"#### Seed {s} ({chosen_variant.capitalize()} Variant)\n")
            f.write(f"- **Augmented Slices** (`aug` in filename, n={diag['aug_count']}): Error Rate = **{diag['aug_error_rate'] * 100:.2f}%** (Accuracy = {diag['aug_accuracy'] * 100:.2f}%)\n")
            f.write(f"- **Non-Augmented Slices** (Original filename, n={diag['non_aug_count']}): Error Rate = **{diag['non_aug_error_rate'] * 100:.2f}%** (Accuracy = {diag['non_aug_accuracy'] * 100:.2f}%)\n")
            f.write(f"- **Per-Class Recall**: {diag['per_class_recall']}\n\n")

        f.write("### 4. Per-Task Epoch Budget E Determination for Stage B\n\n")
        f.write(f"- **3-Epoch Moving Average Analysis**: Reaches within 0.5 pp of maximum at epochs {chosen_runs[0]['budget_E_calculation']['first_epoch_within_0_5_pp']} (Seed 42) and {chosen_runs[1]['budget_E_calculation']['first_epoch_within_0_5_pp']} (Seed 123).\n")
        f.write(f"- **Selected Budget E**: **`E = {final_E}`** epochs (rounded up to nearest multiple of 5).\n\n")

    # ------------------ Plot Confusion Matrix for Chosen Variant (Best Seed) ------------------
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

    return chosen_variant, variant_stats, final_E, chosen_diagnostics


def main() -> None:
    chosen_var, stats, final_E, diags = run_all_joint_experiments()
    logger.info("Stage A Joint Training completed! Selected variant: %s | Budget E: %d", chosen_var, final_E)


if __name__ == "__main__":
    main()

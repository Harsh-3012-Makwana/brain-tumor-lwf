"""Benchmark and profile Custom CNN variants (small, medium, large)."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Insert project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.custom_cnn.factory import build_model
from src.models.utils.profiling import (
    benchmark_latency,
    count_flops,
    count_parameters,
    get_model_size_mb,
)
from src.utils.paths import TABLES_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("model_profiling")

VARIANTS = ["small", "medium", "large"]
HARD_PARAM_LIMIT = 5_000_000


def profile_all_variants() -> List[Dict[str, Any]]:
    """Profile parameters, FLOPs, memory footprint, and latency for all variants."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    results: List[Dict[str, Any]] = []

    for var in VARIANTS:
        logger.info("Profiling variant: %s ...", var)
        model = build_model(variant=var, num_classes=4)

        # 1. Parameter counting
        param_info = count_parameters(model)
        total_p = param_info["total"]
        trainable_p = param_info["trainable"]
        backbone_p = param_info["backbone"]
        head_p = param_info["head"]

        # 2. FLOPs calculation
        flops = count_flops(model, input_size=(1, 3, 150, 150), device="cpu")

        # 3. Model size in MB
        size_mb = get_model_size_mb(model)

        # 4. Latency benchmarks
        logger.info("Measuring CPU latency for %s ...", var)
        cpu_bench = benchmark_latency(
            model, device="cpu", input_size=(1, 3, 150, 150), warmup_runs=20, test_runs=50
        )
        cpu_latency_ms = cpu_bench["mean_ms"]
        cpu_latency_std = cpu_bench["std_ms"]

        gpu_latency_ms = 0.0
        gpu_latency_std = 0.0
        if cuda_available:
            logger.info("Measuring GPU latency for %s on %s ...", var, torch.cuda.get_device_name(0))
            gpu_bench = benchmark_latency(
                model, device="cuda", input_size=(1, 3, 150, 150), warmup_runs=20, test_runs=50
            )
            gpu_latency_ms = gpu_bench["mean_ms"]
            gpu_latency_std = gpu_bench["std_ms"]

        passed_budget = "PASS" if total_p < HARD_PARAM_LIMIT else "FAIL"

        row = {
            "variant": var,
            "total_params": total_p,
            "trainable_params": trainable_p,
            "backbone_params": backbone_p,
            "head_params": head_p,
            "flops": flops,
            "mflops": round(flops / 1e6, 2),
            "model_size_mb": round(size_mb, 2),
            "cpu_latency_ms": round(cpu_latency_ms, 2),
            "cpu_latency_std": round(cpu_latency_std, 2),
            "gpu_latency_ms": round(gpu_latency_ms, 2),
            "gpu_latency_std": round(gpu_latency_std, 2),
            "budget_check (<5M)": passed_budget,
        }
        results.append(row)
        logger.info(
            "%s: %d params (%.2f MB), %.2f MFLOPs, CPU: %.2f ms, GPU: %.2f ms -> %s",
            var.upper(),
            total_p,
            size_mb,
            flops / 1e6,
            cpu_latency_ms,
            gpu_latency_ms,
            passed_budget,
        )

    # Save to CSV
    csv_path = TABLES_DIR / "model_variants.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved CSV table to %s", csv_path)

    # Save to Markdown
    md_path = TABLES_DIR / "model_variants.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Custom Lightweight CNN Model Variants Profiling Report\n\n")
        f.write(f"- **Evaluated Input Resolution**: `3 x 150 x 150` (batch size = 1)\n")
        f.write(f"- **Classification Head**: 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`)\n")
        f.write(f"- **Hard Parameter Cap**: 5,000,000 parameters\n")
        f.write(f"- **CPU Hardware**: Intel Core i7-14650HX CPU\n")
        if cuda_available:
            f.write(f"- **GPU Hardware**: {torch.cuda.get_device_name(0)} (Blackwell `sm_120`)\n\n")
        else:
            f.write("- **GPU Hardware**: None (CPU only)\n\n")

        f.write("| Variant | Total Params | Backbone Params | Head Params | FLOPs (M) | Size (MB) | CPU Latency (ms) | GPU Latency (ms) | Limit Check (<5M) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            f.write(
                f"| **{r['variant'].capitalize()}** | {r['total_params']:,} | "
                f"{r['backbone_params']:,} | {r['head_params']:,} | "
                f"{r['mflops']} M | {r['model_size_mb']:.2f} MB | "
                f"{r['cpu_latency_ms']:.2f} ± {r['cpu_latency_std']:.2f} | "
                f"{r['gpu_latency_ms']:.2f} ± {r['gpu_latency_std']:.2f} | "
                f"**{r['budget_check (<5M)']}** |\n"
            )
        f.write("\n")
    logger.info("Saved Markdown table to %s", md_path)

    return results


def main() -> None:
    profile_all_variants()


if __name__ == "__main__":
    main()

"""Model profiling utilities: parameter counts, FLOPs, disk size, and execution latency."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate total, trainable, and per-submodule parameter counts.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary containing 'total', 'trainable', 'backbone', 'head'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_params = 0
    if hasattr(model, "backbone"):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())

    head_params = 0
    if hasattr(model, "head"):
        head_params = sum(p.numel() for p in model.head.parameters())

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "backbone": backbone_params,
        "head": head_params,
    }


def count_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 150, 150),
    device: str = "cpu",
) -> int:
    """Calculate forward pass FLOPs using fvcore (with thop fallback).

    Args:
        model: PyTorch model.
        input_size: Batch shape (B, C, H, W). Default (1, 3, 150, 150).
        device: Device on which to evaluate FLOPs.

    Returns:
        Total FLOP count as integer.
    """
    model_eval = model.to(device)
    model_eval.eval()
    dummy_input = torch.randn(*input_size, device=device)

    # 1. Attempt fvcore FlopCountAnalysis
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model_eval, dummy_input)
        fca.unsupported_ops_warnings(False)
        return int(fca.total())
    except Exception as e:
        logger.debug("fvcore FLOP counting failed (%s), attempting thop...", e)

    # 2. Fallback to thop
    try:
        import thop
        flops, _ = thop.profile(model_eval, inputs=(dummy_input,), verbose=False)
        return int(flops)
    except Exception as e:
        logger.warning("thop FLOP counting also failed: %s", e)
        return 0


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate in-memory footprint of model weights and buffers in megabytes (MB)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def benchmark_latency(
    model: nn.Module,
    device: str = "cpu",
    input_size: Tuple[int, int, int, int] = (1, 3, 150, 150),
    warmup_runs: int = 20,
    test_runs: int = 100,
) -> Dict[str, float]:
    """Measure inference latency per slice (batch size = 1) on CPU or GPU.

    Args:
        model: PyTorch model.
        device: "cpu" or "cuda".
        input_size: Input tensor dimensions (1, 3, 150, 150).
        warmup_runs: Iterations for thermal/JIT stabilization.
        test_runs: Number of timed executions for mean/std computation.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'.
    """
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=target_device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if target_device.type == "cuda":
            torch.cuda.synchronize()

    times_ms = []

    if target_device.type == "cuda":
        with torch.no_grad():
            for _ in range(test_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
    else:
        with torch.no_grad():
            for _ in range(test_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }

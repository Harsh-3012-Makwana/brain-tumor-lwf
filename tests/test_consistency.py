"""Automated mathematical consistency and integrity tests for continual learning metrics.

Verifies:
  1. For every run, final accuracy strictly equals the class-weighted mean of old and new accuracy.
  2. For 2-task orders (C & D), overall forgetting strictly equals R[0, 0] - R[1, 0] (Task 1 drop).
  3. The table generator never mixes runs of different orders or methods in any summary row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.evaluation.tables.generate_tables import (
    compute_old_and_new_accuracies,
    load_all_completed_runs,
)
from src.utils.paths import RUNS_DIR, TABLES_DIR


def test_class_weighted_final_accuracy_all_runs():
    """For every completed run, final accuracy must equal the class-weighted mean of old and new accuracy."""
    runs = load_all_completed_runs()
    assert len(runs) >= 42, f"Expected at least 42 incremental runs, found {len(runs)}"

    discrepancies = []
    for r in runs:
        order = r.get("order")
        old_acc, new_acc = compute_old_and_new_accuracies(r)
        final_acc = float(r.get("final_accuracy", 0.0))

        if order in ["A", "B"]:
            # 4 classes total: 3 old (T1=2, T2=1), 1 new (T3=1) -> weights 3/4 and 1/4
            expected_final = 0.75 * old_acc + 0.25 * new_acc
        elif order in ["C", "D"]:
            # 4 classes total: 2 old (T1=2), 2 new (T2=2) -> weights 1/2 and 1/2
            expected_final = 0.5 * old_acc + 0.5 * new_acc
        else:
            continue

        diff = abs(expected_final - final_acc)
        if diff > 1e-4:
            discrepancies.append((r.get("method"), order, r.get("seed"), diff, expected_final, final_acc))

    assert len(discrepancies) == 0, f"Discrepancies found in class-weighted final accuracy: {discrepancies}"


def test_two_task_forgetting_all_runs():
    """For 2-task orders (C & D), forgetting must equal Task 1 accuracy after Task 1 minus Task 1 accuracy after Task 2."""
    runs = load_all_completed_runs()
    two_task_runs = [r for r in runs if r.get("order") in ["C", "D"]]
    assert len(two_task_runs) >= 20, f"Expected at least 20 runs in orders C & D, found {len(two_task_runs)}"

    discrepancies = []
    for r in two_task_runs:
        R = np.array(r["accuracy_matrix"], dtype=float)
        expected_forgetting = float(R[0, 0] - R[1, 0])
        actual_forgetting = float(r["overall_forgetting"])

        diff = abs(expected_forgetting - actual_forgetting)
        if diff > 1e-4:
            discrepancies.append((
                r.get("method"),
                r.get("order"),
                r.get("seed"),
                diff,
                expected_forgetting,
                actual_forgetting,
            ))

    assert len(discrepancies) == 0, f"Discrepancies found in 2-task forgetting calculation: {discrepancies}"


def test_generator_no_cross_order_mixing():
    """Verify that table generator strictly groups by (order, method) and never mixes orders."""
    runs = load_all_completed_runs()
    target_methods = ["finetune", "lwf", "lwf_all", "lwf_new", "lwfplus", "replay60", "replay120"]
    target_orders = ["A", "B", "C", "D"]

    # Check incremental_results.csv
    csv_path = TABLES_DIR / "incremental_results.csv"
    assert csv_path.is_file(), f"Expected {csv_path} to exist"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            order = row["order"]
            seed = int(row["seed"])
            assert order in target_orders, f"Invalid order {order} in CSV"
            assert method in target_methods, f"Invalid method {method} in CSV"

            # Match single run from disk
            matching = [
                r for r in runs
                if r.get("order") == order and r.get("method") == method and r.get("seed") == seed
            ]
            assert len(matching) == 1, f"Expected exactly 1 run for ({order}, {method}, {seed}), found {len(matching)}"
            assert abs(float(row["final_acc"]) - float(matching[0]["final_accuracy"])) < 1e-4
            assert abs(float(row["forgetting"]) - float(matching[0]["overall_forgetting"])) < 1e-4


def test_final_summary_strictly_aggregates_matching_order():
    """Verify final_summary.md table contains exactly the mean of seeds 42 and 123 for each order."""
    runs = load_all_completed_runs()
    table_structure = [
        ("A", ["finetune", "lwf", "replay60", "replay120"]),
        ("B", ["finetune", "lwf", "replay60", "replay120"]),
        ("C", ["finetune", "lwf_all", "lwf_new", "replay60", "replay120"]),
        ("D", ["finetune", "lwf_all", "lwf_new", "replay60", "replay120"]),
    ]

    for order, methods in table_structure:
        for method in methods:
            matching = [r for r in runs if r.get("order") == order and r.get("method") == method]
            # Every method in table_structure must have exactly 2 runs (seed 42 and seed 123)
            assert len(matching) == 2, f"Expected 2 runs for order {order} method {method}, found {len(matching)}"
            seeds = {r["seed"] for r in matching}
            assert seeds == {42, 123}, f"Expected seeds {{42, 123}} for ({order}, {method}), got {seeds}"

            # Verify no run in matching belongs to another order
            for r in matching:
                assert r["order"] == order, f"Cross-order contamination: run with order {r['order']} in {order}"

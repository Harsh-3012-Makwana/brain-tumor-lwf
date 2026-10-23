"""Unit tests for Phase 1 Data Engineering pipeline."""

import csv
import filecmp
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset

from src.data.datasets.mri_dataset import GLOBAL_LABEL_MAP, MRIDataset
from src.data.splitting.split import (
    create_stratified_train_val_split,
    load_audit_records,
)
from src.data.tasks.builder import (
    TASK_ORDERS,
    get_joint_datasets,
    get_task_datasets,
    sample_replay_buffer,
)
from src.utils.paths import SPLITS_DIR


def test_fixed_labels():
    """Verify that global label IDs strictly match the frozen specification."""
    expected = {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3,
    }
    assert GLOBAL_LABEL_MAP == expected, f"Label map mismatch: {GLOBAL_LABEL_MAP}"


def test_train_val_stratification():
    """Verify val split is ~15% per class with no class missing."""
    for seed in [42, 123]:
        csv_path = SPLITS_DIR / f"train_val_seed{seed}.csv"
        assert csv_path.exists(), f"Split CSV {csv_path} does not exist"

        counts = Counter()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                counts[(row["split"], int(row["global_label"]))] += 1

        for cls_id in range(4):
            val_cnt = counts[("val", cls_id)]
            train_cnt = counts[("train", cls_id)]
            total = val_cnt + train_cnt
            assert total == 1400, f"Expected 1400 total per class, got {total}"
            val_fraction = val_cnt / total
            assert 0.14 <= val_fraction <= 0.16, (
                f"Seed {seed}, Class {cls_id}: Val fraction {val_fraction:.3f} outside [0.14, 0.16]"
            )


def test_no_leakage_between_splits():
    """Verify train, val, and test splits have zero overlapping file paths."""
    test_csv = SPLITS_DIR / "test.csv"
    test_paths = set()
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_paths.add(row["path"])

    assert len(test_paths) == 1600, f"Expected 1600 test paths, got {len(test_paths)}"

    for seed in [42, 123]:
        train_val_csv = SPLITS_DIR / f"train_val_seed{seed}.csv"
        train_paths = set()
        val_paths = set()
        with open(train_val_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == "train":
                    train_paths.add(row["path"])
                elif row["split"] == "val":
                    val_paths.add(row["path"])

        # 1. No overlap between train and val
        train_val_intersect = train_paths.intersection(val_paths)
        assert len(train_val_intersect) == 0, (
            f"Seed {seed}: Found {len(train_val_intersect)} paths overlapping between train and val!"
        )

        # 2. No overlap with test set
        train_test_intersect = train_paths.intersection(test_paths)
        assert len(train_test_intersect) == 0, (
            f"Seed {seed}: Found {len(train_test_intersect)} paths leaking into test set!"
        )
        val_test_intersect = val_paths.intersection(test_paths)
        assert len(val_test_intersect) == 0, (
            f"Seed {seed}: Found {len(val_test_intersect)} val paths leaking into test set!"
        )


def test_split_determinism():
    """Verify that splitting with the same seed produces identical allocations."""
    records = load_audit_records()
    train_records = [r for r in records if r["split"] == "Training"]

    split1 = create_stratified_train_val_split(train_records, seed=42)
    split2 = create_stratified_train_val_split(train_records, seed=42)

    assert len(split1) == len(split2)
    for r1, r2 in zip(split1, split2):
        assert r1["path"] == r2["path"]
        assert r1["split"] == r2["split"]
        assert r1["global_label"] == r2["global_label"]


def test_task_builder_orders():
    """Verify task builder class orders and cumulative sizes for Orders A and B."""
    assert TASK_ORDERS["A"] == [[0, 1], [3], [2]], "Order A mismatch"
    assert TASK_ORDERS["B"] == [[2, 3], [0], [1]], "Order B mismatch"

    # Test Order A
    tasks_a = get_task_datasets("A", seed=42)
    assert len(tasks_a) == 3
    assert tasks_a[0].new_classes == [0, 1]
    assert tasks_a[0].head_size == 2
    assert tasks_a[1].new_classes == [3]
    assert tasks_a[1].head_size == 3
    assert tasks_a[2].new_classes == [2]
    assert tasks_a[2].head_size == 4

    # Test Order B
    tasks_b = get_task_datasets("B", seed=42)
    assert len(tasks_b) == 3
    assert tasks_b[0].new_classes == [2, 3]
    assert tasks_b[0].head_size == 2
    assert tasks_b[1].new_classes == [0]
    assert tasks_b[1].head_size == 3
    assert tasks_b[2].new_classes == [1]
    assert tasks_b[2].head_size == 4


def test_dataset_item_shape():
    """Verify MRIDataset returns (3, 150, 150) tensors and valid labels."""
    test_csv = SPLITS_DIR / "test.csv"
    ds = MRIDataset(test_csv)
    assert len(ds) == 1600

    img_tensor, label, path_str = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 150, 150), f"Expected (3, 150, 150), got {img_tensor.shape}"
    assert img_tensor.dtype in [torch.float32, torch.float64]
    assert isinstance(label, int)
    assert 0 <= label <= 3
    assert isinstance(path_str, str)


def test_replay_buffer_size_and_balance():
    """Verify replay buffer has exactly 60 or 120 images, balanced across old classes."""
    train_val_csv = SPLITS_DIR / "train_val_seed42.csv"
    # Old classes from Order A Task 1: {glioma: 0, meningioma: 1}
    ds_t1 = MRIDataset(train_val_csv, split="train", allowed_classes=[0, 1])

    # Test buffer size 60 (30 per class)
    buf60 = sample_replay_buffer(ds_t1, buffer_size=60, seed=42)
    assert len(buf60) == 60
    labels60 = [buf60[i][1] for i in range(len(buf60))]
    counts60 = Counter(labels60)
    assert counts60[0] == 30
    assert counts60[1] == 30

    # Test buffer size 120 (60 per class)
    buf120 = sample_replay_buffer(ds_t1, buffer_size=120, seed=42)
    assert len(buf120) == 120
    labels120 = [buf120[i][1] for i in range(len(buf120))]
    counts120 = Counter(labels120)
    assert counts120[0] == 60
    assert counts120[1] == 60

    # After Task 2 (3 classes: 0, 1, 3)
    ds_t1_t2 = MRIDataset(train_val_csv, split="train", allowed_classes=[0, 1, 3])
    buf60_3cls = sample_replay_buffer(ds_t1_t2, buffer_size=60, seed=42)
    assert len(buf60_3cls) == 60
    labels60_3 = [buf60_3cls[i][1] for i in range(len(buf60_3cls))]
    counts60_3 = Counter(labels60_3)
    assert counts60_3[0] == 20
    assert counts60_3[1] == 20
    assert counts60_3[3] == 20

"""Dataset splitting module: stratified train/val split with duplicate cluster isolation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

# Insert project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils.paths import PROCESSED_DATA_DIR, SPLITS_DIR, TABLES_DIR

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLOBAL_LABEL_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}


class UnionFind:
    """Disjoint-set data structure to cluster duplicate and near-duplicate images."""

    def __init__(self, elements: List[str]) -> None:
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def get_clusters(self) -> List[List[str]]:
        clusters = defaultdict(list)
        for el in self.parent:
            clusters[self.find(el)].append(el)
        return list(clusters.values())


def load_audit_records() -> List[Dict[str, Any]]:
    """Load image audit records from results/tables/data_audit.csv."""
    audit_csv = TABLES_DIR / "data_audit.csv"
    if not audit_csv.is_file():
        raise FileNotFoundError(
            f"Audit file not found at {audit_csv}. Run data audit first."
        )

    records: List[Dict[str, Any]] = []
    with open(audit_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "split": row["split"],
                "class": row["class"],
                "label_id": int(row["label_id"]),
                "filename": row["filename"],
                "rel_path": row["rel_path"],
                "sha256": row["sha256"],
                "phash": row["phash"],
            })
    return records


def cluster_near_duplicates_in_train(
    train_records: List[Dict[str, Any]], phash_threshold: int = 4
) -> List[List[Dict[str, Any]]]:
    """Group near-duplicate images in training set so they never split across train and val.

    Clustering is strictly within each class.
    """
    path_to_rec = {r["rel_path"]: r for r in train_records}
    all_clusters: List[List[Dict[str, Any]]] = []

    # Group by class
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in train_records:
        by_class[r["class"]].append(r)

    for cls_name, cls_records in by_class.items():
        paths = [r["rel_path"] for r in cls_records]
        uf = UnionFind(paths)

        # 1. Exact SHA-256 grouping
        sha_groups = defaultdict(list)
        for r in cls_records:
            sha_groups[r["sha256"]].append(r["rel_path"])
        for group in sha_groups.values():
            for i in range(1, len(group)):
                uf.union(group[0], group[i])

        # 2. pHash distance grouping via vectorized comparison
        hashes = np.array([int(r["phash"], 16) for r in cls_records], dtype=np.uint64)
        if len(hashes) > 0:
            dist_matrix = np.bitwise_count(hashes[:, None] ^ hashes[None, :])
            i_idx, j_idx = np.triu_indices(len(hashes), k=1)
            mask = dist_matrix[i_idx, j_idx] <= phash_threshold
            for i, j in zip(i_idx[mask], j_idx[mask]):
                uf.union(paths[i], paths[j])

        cls_clusters = uf.get_clusters()
        for c in cls_clusters:
            all_clusters.append([path_to_rec[p] for p in c])

    return all_clusters


def create_stratified_train_val_split(
    train_records: List[Dict[str, Any]],
    val_fraction: float = 0.15,
    seed: int = 42,
    phash_threshold: int = 4,
) -> List[Dict[str, Any]]:
    """Carve a stratified ~15% validation split keeping duplicate clusters intact."""
    clusters = cluster_near_duplicates_in_train(train_records, phash_threshold=phash_threshold)

    # Group clusters by class
    clusters_by_class: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    for c in clusters:
        cls_name = c[0]["class"]
        clusters_by_class[cls_name].append(c)

    rng = random.Random(seed)
    split_rows: List[Dict[str, Any]] = []

    for cls_name in CLASS_NAMES:
        cls_clusters = clusters_by_class[cls_name]
        # Deterministically shuffle clusters
        rng.shuffle(cls_clusters)

        total_cls_images = sum(len(c) for c in cls_clusters)
        target_val = int(round(total_cls_images * val_fraction))

        curr_val = 0
        for c in cls_clusters:
            c_size = len(c)
            # Greedy allocation: if adding keeps us near target, put in val
            if (curr_val + c_size <= target_val) or (curr_val == 0 and c_size > target_val):
                assigned_split = "val"
                curr_val += c_size
            else:
                assigned_split = "train"

            for r in c:
                # Path pointing to processed image
                proc_path = f"data/processed/{r['rel_path']}"
                split_rows.append({
                    "path": proc_path,
                    "class_name": r["class"],
                    "global_label": r["label_id"],
                    "split": assigned_split,
                })

    return split_rows


def create_test_split(test_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create test split entries for all held-out test images."""
    rows: List[Dict[str, Any]] = []
    for r in sorted(test_records, key=lambda x: x["rel_path"]):
        proc_path = f"data/processed/{r['rel_path']}"
        rows.append({
            "path": proc_path,
            "class_name": r["class"],
            "global_label": r["label_id"],
            "split": "test",
        })
    return rows


def write_split_csv(rows: List[Dict[str, Any]], out_path: Path) -> str:
    """Save split CSV and return its SHA-256 checksum."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "class_name", "global_label", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    hasher = hashlib.sha256()
    with open(out_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_all_splits(seeds: List[int] = [42, 123]) -> Dict[str, str]:
    """Generate train/val splits for all specified seeds and the test split."""
    records = load_audit_records()
    train_records = [r for r in records if r["split"] == "Training"]
    test_records = [r for r in records if r["split"] == "Testing"]

    checksums: Dict[str, str] = {}

    # 1. Test split
    test_rows = create_test_split(test_records)
    test_csv_path = SPLITS_DIR / "test.csv"
    checksums["test.csv"] = write_split_csv(test_rows, test_csv_path)
    logger.info("Saved %s (%d test images)", test_csv_path, len(test_rows))

    # 2. Train/Val splits for each seed
    for seed in seeds:
        split_rows = create_stratified_train_val_split(
            train_records, val_fraction=0.15, seed=seed, phash_threshold=4
        )
        csv_name = f"train_val_seed{seed}.csv"
        csv_path = SPLITS_DIR / csv_name
        checksums[csv_name] = write_split_csv(split_rows, csv_path)

        train_cnt = sum(1 for r in split_rows if r["split"] == "train")
        val_cnt = sum(1 for r in split_rows if r["split"] == "val")
        logger.info(
            "Saved %s: %d train, %d val (Total: %d, Val fraction: %.2f%%)",
            csv_path,
            train_cnt,
            val_cnt,
            len(split_rows),
            (val_cnt / len(split_rows)) * 100,
        )

    # 3. Save checksums JSON
    checksums_path = SPLITS_DIR / "checksums.json"
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    logger.info("Saved split checksums to %s", checksums_path)

    return checksums


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    generate_all_splits()

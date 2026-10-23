"""Dataset integrity, distribution, duplicate, and cross-split leakage audit module."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from tqdm import tqdm

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GLOBAL_LABEL_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(image_gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Compute Perceptual Hash (pHash) using discrete cosine transform (DCT).

    Args:
        image_gray: 2D uint8 grayscale image array.
        hash_size: Output binary hash grid width/height (default 8 -> 64-bit hash).
        highfreq_factor: Factor for initial downscaling resolution (default 4 -> 32x32).

    Returns:
        Hexadecimal string representation of the 64-bit hash.
    """
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(image_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # Compute 2D DCT
    dct_block = dct(dct(resized.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho")
    # Extract low frequency coefficients (top-left, omitting DC component)
    dct_lowfreq = dct_block[:hash_size, :hash_size]
    # Median of low-freq components excluding DC
    med = np.median(dct_lowfreq.flatten()[1:])
    # Generate binary bitstring
    bitstring = (dct_lowfreq > med).flatten()
    # Convert to hex string
    hex_str = "".join(f"{int(b):01b}" for b in bitstring)
    return f"{int(hex_str, 2):016x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Compute bitwise Hamming distance between two 16-character hex hashes."""
    x = int(h1, 16) ^ int(h2, 16)
    return bin(x).count("1")


def scan_dataset(raw_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan raw directory, read dimensions, test decodability, and compute hashes."""
    records: List[Dict[str, Any]] = []
    corrupted_files: List[str] = []

    for split in ["Training", "Testing"]:
        split_dir = raw_dir / split
        if not split_dir.is_dir():
            continue

        for cls_name in CLASS_NAMES:
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                continue

            for file_path in sorted(cls_dir.iterdir()):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue

                rel_path = file_path.relative_to(raw_dir)
                # 1. PIL verification
                try:
                    with Image.open(file_path) as pil_img:
                        pil_img.verify()
                except Exception as e:
                    logger.warning("Corrupted PIL image: %s (%s)", file_path, e)
                    corrupted_files.append(str(rel_path))
                    continue

                # 2. OpenCV decoding verification
                cv_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if cv_img is None:
                    logger.warning("Unreadable OpenCV image: %s", file_path)
                    corrupted_files.append(str(rel_path))
                    continue

                h, w = cv_img.shape[:2]
                channels = cv_img.shape[2] if len(cv_img.shape) == 3 else 1

                # Grayscale version for perceptual hashing
                if channels == 3:
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                elif channels == 4:
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv_img

                sha256 = compute_sha256(file_path)
                phash = compute_phash(gray)

                records.append({
                    "split": split,
                    "class": cls_name,
                    "label_id": GLOBAL_LABEL_MAP[cls_name],
                    "filename": file_path.name,
                    "rel_path": str(rel_path).replace("\\", "/"),
                    "abs_path": str(file_path.resolve()),
                    "height": h,
                    "width": w,
                    "channels": channels,
                    "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
                    "file_size_bytes": file_path.stat().st_size,
                    "sha256": sha256,
                    "phash": phash,
                })

    return records, corrupted_files


def detect_duplicates_and_leakage(
    records: List[Dict[str, Any]], phash_threshold: int = 4
) -> Dict[str, Any]:
    """Identify exact and perceptual near-duplicates within splits and across train vs test."""
    exact_hash_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in records:
        exact_hash_to_records[r["sha256"]].append(r)

    # 1. Exact duplicates
    exact_within_train: List[List[str]] = []
    exact_within_test: List[List[str]] = []
    exact_cross_leakage: List[Dict[str, Any]] = []

    for sha, group in exact_hash_to_records.items():
        if len(group) > 1:
            splits_present = set(g["split"] for g in group)
            paths = [g["rel_path"] for g in group]
            if "Training" in splits_present and "Testing" in splits_present:
                train_paths = [g["rel_path"] for g in group if g["split"] == "Training"]
                test_paths = [g["rel_path"] for g in group if g["split"] == "Testing"]
                classes = list(set(g["class"] for g in group))
                exact_cross_leakage.append({
                    "sha256": sha,
                    "train_images": train_paths,
                    "test_images": test_paths,
                    "classes": classes,
                })
            elif "Training" in splits_present:
                exact_within_train.append(paths)
            elif "Testing" in splits_present:
                exact_within_test.append(paths)

    # 2. Vectorized pHash comparison
    train_records = [r for r in records if r["split"] == "Training"]
    test_records = [r for r in records if r["split"] == "Testing"]

    train_hashes = np.array([int(r["phash"], 16) for r in train_records], dtype=np.uint64)
    test_hashes = np.array([int(r["phash"], 16) for r in test_records], dtype=np.uint64)

    # Check cross leakage (train vs test)
    near_cross_leakage: List[Dict[str, Any]] = []
    if len(train_hashes) > 0 and len(test_hashes) > 0:
        # matrix of size (N_train, N_test)
        cross_dist = np.bitwise_count(train_hashes[:, None] ^ test_hashes[None, :])
        # Find pairs where dist <= threshold and not exact duplicate
        matches = np.argwhere(cross_dist <= phash_threshold)
        for i, j in matches:
            tr = train_records[i]
            te = test_records[j]
            if tr["sha256"] != te["sha256"]:
                near_cross_leakage.append({
                    "train_image": tr["rel_path"],
                    "test_image": te["rel_path"],
                    "train_class": tr["class"],
                    "test_class": te["class"],
                    "hamming_dist": int(cross_dist[i, j]),
                })

    # Near duplicates within train (upper triangle)
    near_within_train: List[Tuple[str, str, int]] = []
    if len(train_hashes) > 0:
        train_dist = np.bitwise_count(train_hashes[:, None] ^ train_hashes[None, :])
        i_idx, j_idx = np.triu_indices(len(train_hashes), k=1)
        mask = train_dist[i_idx, j_idx] <= phash_threshold
        for i, j in zip(i_idx[mask], j_idx[mask]):
            if train_records[i]["sha256"] != train_records[j]["sha256"]:
                near_within_train.append((
                    train_records[i]["rel_path"],
                    train_records[j]["rel_path"],
                    int(train_dist[i, j]),
                ))

    # Near duplicates within test (upper triangle)
    near_within_test: List[Tuple[str, str, int]] = []
    if len(test_hashes) > 0:
        test_dist = np.bitwise_count(test_hashes[:, None] ^ test_hashes[None, :])
        i_idx, j_idx = np.triu_indices(len(test_hashes), k=1)
        mask = test_dist[i_idx, j_idx] <= phash_threshold
        for i, j in zip(i_idx[mask], j_idx[mask]):
            if test_records[i]["sha256"] != test_records[j]["sha256"]:
                near_within_test.append((
                    test_records[i]["rel_path"],
                    test_records[j]["rel_path"],
                    int(test_dist[i, j]),
                ))

    return {
        "exact_within_train": exact_within_train,
        "exact_within_test": exact_within_test,
        "exact_cross_leakage": exact_cross_leakage,
        "near_within_train": near_within_train,
        "near_within_train_count": len(near_within_train),
        "near_within_test": near_within_test,
        "near_within_test_count": len(near_within_test),
        "near_cross_leakage": near_cross_leakage,
    }


def generate_audit_reports(
    records: List[Dict[str, Any]],
    corrupted_files: List[str],
    dup_results: Dict[str, Any],
    output_table_dir: Path,
    output_fig_dir: Path,
) -> None:
    """Export data audit reports in CSV and Markdown formats, and generate distribution plots."""
    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    # CSV record dump
    csv_path = output_table_dir / "data_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "class", "label_id", "filename", "rel_path",
                "height", "width", "channels", "aspect_ratio",
                "file_size_bytes", "sha256", "phash"
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    # Class distribution aggregation
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    resolutions = [(r["width"], r["height"]) for r in records]

    for r in records:
        counts[r["split"]][r["class"]] += 1

    train_total = sum(counts["Training"].values())
    test_total = sum(counts["Testing"].values())
    grand_total = train_total + test_total

    # Markdown summary report
    md_path = output_table_dir / "data_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Brain Tumor MRI Dataset Audit Report\n\n")
        f.write(f"- **Total Scanned Images**: {grand_total}\n")
        f.write(f"- **Corrupted / Unreadable Images**: {len(corrupted_files)}\n")
        f.write(f"- **Training Split Total**: {train_total}\n")
        f.write(f"- **Testing Split Total**: {test_total}\n\n")

        f.write("## 1. Class Distribution\n\n")
        f.write("| Split | Glioma | Meningioma | Notumor | Pituitary | Total |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(
            f"| **Training** | {counts['Training']['glioma']} | "
            f"{counts['Training']['meningioma']} | {counts['Training']['notumor']} | "
            f"{counts['Training']['pituitary']} | {train_total} |\n"
        )
        f.write(
            f"| **Testing** | {counts['Testing']['glioma']} | "
            f"{counts['Testing']['meningioma']} | {counts['Testing']['notumor']} | "
            f"{counts['Testing']['pituitary']} | {test_total} |\n"
        )
        f.write(
            f"| **Overall** | {counts['Training']['glioma'] + counts['Testing']['glioma']} | "
            f"{counts['Training']['meningioma'] + counts['Testing']['meningioma']} | "
            f"{counts['Training']['notumor'] + counts['Testing']['notumor']} | "
            f"{counts['Training']['pituitary'] + counts['Testing']['pituitary']} | {grand_total} |\n\n"
        )

        f.write("## 2. Image Resolutions & Geometry\n\n")
        widths = [r["width"] for r in records]
        heights = [r["height"] for r in records]
        f.write(f"- **Min Resolution**: {min(widths)} x {min(heights)}\n")
        f.write(f"- **Max Resolution**: {max(widths)} x {max(heights)}\n")
        f.write(f"- **Mean Resolution**: {np.mean(widths):.1f} x {np.mean(heights):.1f}\n")
        f.write(f"- **Aspect Ratio Range**: {min(r['aspect_ratio'] for r in records)} to {max(r['aspect_ratio'] for r in records)}\n\n")

        f.write("## 3. Duplicate Analysis & Cross-Split Leakage\n\n")
        f.write(f"- **Exact Duplicates within Training**: {len(dup_results['exact_within_train'])} groups\n")
        f.write(f"- **Exact Duplicates within Testing**: {len(dup_results['exact_within_test'])} groups\n")
        f.write(f"- **Exact Duplicates ACROSS Train vs Test (Direct Leakage)**: {len(dup_results['exact_cross_leakage'])} pairs/groups\n")
        f.write(f"- **Near-Duplicates ACROSS Train vs Test (pHash Hamming <= 4)**: {len(dup_results['near_cross_leakage'])} pairs\n\n")

        if dup_results["exact_cross_leakage"]:
            f.write("### Exact Cross-Split Leakage Details\n\n")
            f.write("| SHA256 (Prefix) | Train File(s) | Test File(s) | Classes |\n")
            f.write("|:---|:---|:---|:---|\n")
            for item in dup_results["exact_cross_leakage"][:50]:
                f.write(f"| `{item['sha256'][:10]}` | {', '.join(item['train_images'])} | {', '.join(item['test_images'])} | {', '.join(item['classes'])} |\n")
            if len(dup_results["exact_cross_leakage"]) > 50:
                f.write(f"| ... | *and {len(dup_results['exact_cross_leakage']) - 50} more* | | |\n")
            f.write("\n")

        if dup_results["near_cross_leakage"]:
            f.write("### Top Near-Duplicate Cross-Split Pairs (Hamming <= 4)\n\n")
            f.write("| Train Image | Test Image | Train Class | Test Class | Hamming Distance |\n")
            f.write("|:---|:---|:---:|:---:|:---:|\n")
            for item in dup_results["near_cross_leakage"][:50]:
                f.write(f"| {item['train_image']} | {item['test_image']} | {item['train_class']} | {item['test_class']} | {item['hamming_dist']} |\n")
            if len(dup_results["near_cross_leakage"]) > 50:
                f.write(f"| ... | *and {len(dup_results['near_cross_leakage']) - 50} more* | | | |\n")
            f.write("\n")

    # Plot class distribution figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    train_vals = [counts["Training"][c] for c in CLASS_NAMES]
    test_vals = [counts["Testing"][c] for c in CLASS_NAMES]

    ax.bar(x - width / 2, train_vals, width, label="Training", color="#2b5c8f")
    ax.bar(x + width / 2, test_vals, width, label="Testing", color="#d95f02")

    ax.set_ylabel("Image Count")
    ax.set_title("Brain Tumor MRI Dataset: Class Distribution by Split")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(CLASS_NAMES)):
        ax.text(x[i] - width / 2, train_vals[i] + 20, str(train_vals[i]), ha="center", fontsize=9)
        ax.text(x[i] + width / 2, test_vals[i] + 20, str(test_vals[i]), ha="center", fontsize=9)

    fig.tight_layout()
    fig_path = output_fig_dir / "class_distribution.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved data audit report to %s and figure to %s", md_path, fig_path)

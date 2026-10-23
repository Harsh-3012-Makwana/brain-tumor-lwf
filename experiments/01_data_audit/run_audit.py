"""Runner script for Phase 1 Data Audit."""

import logging
import sys
from pathlib import Path

# Insert project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit.audit import (
    detect_duplicates_and_leakage,
    generate_audit_reports,
    scan_dataset,
)
from src.utils.paths import FIGURES_DIR, RAW_DATA_DIR, TABLES_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("run_audit")


def main() -> None:
    logger.info("Starting Dataset Audit on %s ...", RAW_DATA_DIR)
    records, corrupted = scan_dataset(RAW_DATA_DIR)
    logger.info("Scanned %d valid images, %d corrupted images.", len(records), len(corrupted))

    logger.info("Computing exact and near-duplicate leakage...")
    dup_results = detect_duplicates_and_leakage(records, phash_threshold=4)

    logger.info("Generating reports in %s and %s ...", TABLES_DIR, FIGURES_DIR)
    generate_audit_reports(records, corrupted, dup_results, TABLES_DIR, FIGURES_DIR)

    logger.info(
        "Data audit complete. Found %d exact cross-split leakage instances and %d near-leakage pairs.",
        len(dup_results["exact_cross_leakage"]),
        len(dup_results["near_cross_leakage"]),
    )


if __name__ == "__main__":
    main()

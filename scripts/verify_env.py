"""Environment, dependencies, and configuration sanity check script."""

from __future__ import annotations

import sys
import functools
print = functools.partial(print, flush=True)
import platform
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> bool:
    print("=" * 70)
    print("PHASE 0: ENVIRONMENT, HARDWARE & REPRODUCIBILITY SANITY VERIFICATION")
    print("=" * 70)

    # 1. System & Python
    print(f"\n[1] Python & OS Environment:")
    print(f"  - Python Executable : {sys.executable}")
    print(f"  - Python Version    : {sys.version.split()[0]}")
    print(f"  - Platform / OS     : {platform.platform()}")
    print(f"  - Architecture      : {platform.machine()}")

    # 2. Hardware & PyTorch CUDA
    print(f"\n[2] PyTorch & Hardware Acceleration:")
    try:
        import torch

        print(f"  - PyTorch Version   : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  - CUDA Available    : {cuda_avail}")
        if cuda_avail:
            print(f"  - CUDA Version      : {torch.version.cuda}")
            print(f"  - Device Count      : {torch.cuda.device_count()}")
            print(f"  - Active GPU Device : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  - Total GPU Memory  : {props.total_memory / (1024**2):.1f} MB (~{props.total_memory / (1024**3):.1f} GB)")
        else:
            print("  - WARNING: CUDA is not available. Falling back to CPU.")
    except ImportError as e:
        print(f"  - FAILED to import torch: {e}")
        return False

    # 3. Core Libraries Import Verification
    print(f"\n[3] Core Libraries Import Verification:")
    packages = [
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("opencv-python", "cv2"),
        ("albumentations", "albumentations"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("optuna", "optuna"),
        ("tensorboard", "tensorboard"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("thop", "thop"),
        ("fvcore", "fvcore"),
        ("onnx", "onnx"),
        ("gradio", "gradio"),
    ]

    all_imports_passed = True
    for pkg_name, import_name in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  - [OK] {pkg_name:<16} (import: {import_name:<14} | version: {ver})")
        except ImportError as e:
            print(f"  - [MISSING] {pkg_name:<16} (import: {import_name:<14} | error: {e})")
            all_imports_passed = False

    # 4. Reproducibility & Seed Utility Check
    print(f"\n[4] Reproducibility Utility Check:")
    from src.utils.seed import set_seed

    s1 = set_seed(42, deterministic=True)
    draw_1 = torch.randn(3).tolist()
    s2 = set_seed(42, deterministic=True)
    draw_2 = torch.randn(3).tolist()

    if draw_1 == draw_2:
        print(f"  - [PASS] Seed 42 determinism check passed (identical tensor draws).")
    else:
        print(f"  - [FAIL] Seed determinism failed: {draw_1} != {draw_2}")
        return False

    # 5. Configuration Loading Check
    print(f"\n[5] Config Loader & base.yaml Validation:")
    from src.utils.config import load_yaml
    from src.utils.paths import CONFIGS_DIR

    base_cfg_path = CONFIGS_DIR / "base.yaml"
    if not base_cfg_path.is_file():
        print(f"  - [FAIL] base.yaml not found at {base_cfg_path}")
        return False

    cfg = load_yaml(base_cfg_path)
    print(f"  - [PASS] base.yaml parsed successfully.")
    print(f"    * Project Name     : {cfg.get('project', {}).get('name')}")
    print(f"    * Seeds            : {cfg.get('seeds')}")
    print(f"    * Image Resolution : {cfg.get('data', {}).get('image_size')}")
    print(f"    * Hard Param Limit : {cfg.get('model', {}).get('hard_param_limit')}")
    print(f"    * LwF Temperature  : {cfg.get('incremental', {}).get('lwf', {}).get('temperature')}")
    print(f"    * LwF Lambda       : {cfg.get('incremental', {}).get('lwf', {}).get('lambda_distill')}")

    # 6. Repository Scaffold Check
    print(f"\n[6] Repository Directory Scaffold:")
    required_dirs = [
        "configs", "data/raw", "data/interim", "data/processed", "data/splits",
        "src/data", "src/models", "src/incremental", "src/training", "src/evaluation",
        "experiments", "results/runs", "results/checkpoints", "results/logs",
        "notebooks", "tests", "docs", "scripts", "assets"
    ]
    all_dirs_ok = True
    for d in required_dirs:
        dir_path = PROJECT_ROOT / d
        if dir_path.is_dir():
            print(f"  - [OK] {d}")
        else:
            print(f"  - [MISSING] {d}")
            all_dirs_ok = False

    print("\n" + "=" * 70)
    if all_imports_passed and all_dirs_ok:
        print("ALL PHASE 0 SANITY CHECKS PASSED SUCCESSFULLY.")
    else:
        print("SOME CHECKS REQUIRE ATTENTION (see above logs).")
    print("=" * 70 + "\n")

    return all_imports_passed and all_dirs_ok


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)

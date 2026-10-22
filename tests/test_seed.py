"""Unit tests for reproducibility and seed-setting utilities."""

import random
import numpy as np
import pytest

from src.utils.seed import set_seed


def test_python_and_numpy_seed_determinism():
    """Verify that set_seed ensures reproducible random draws across Python and NumPy."""
    set_seed(42, deterministic=True)
    py_rand_1 = [random.random() for _ in range(5)]
    np_rand_1 = np.random.rand(5).tolist()

    set_seed(42, deterministic=True)
    py_rand_2 = [random.random() for _ in range(5)]
    np_rand_2 = np.random.rand(5).tolist()

    assert py_rand_1 == py_rand_2, "Python random values must match for identical seed."
    assert np.allclose(np_rand_1, np_rand_2), "NumPy random values must match for identical seed."


def test_pytorch_seed_determinism():
    """Verify that PyTorch tensors produce identical pseudo-random numbers with the same seed."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed in test environment yet.")

    set_seed(123, deterministic=True)
    t1 = torch.randn(3, 3)

    set_seed(123, deterministic=True)
    t2 = torch.randn(3, 3)

    assert torch.allclose(t1, t2), "PyTorch random tensors must match for identical seed."


def test_different_seeds_produce_different_draws():
    """Verify that different seeds produce distinct outputs."""
    set_seed(42)
    a = random.random()

    set_seed(123)
    b = random.random()

    assert a != b, "Different seeds should produce different random values."

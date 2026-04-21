"""Tests for pyinfercnv.kernels.smooth_tail_numba."""
from __future__ import annotations

import numpy as np

from pyinfercnv.kernels.smooth_tail_numba import smooth_tail_inplace


def test_short_chromosome_no_op():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = x.copy()
    smooth_tail_inplace(out, window_length=11)
    assert out.shape == x.shape


def test_interior_values_unchanged():
    """Tail smoothing only touches positions within tail (= (w-1)/2) of each end."""
    x = np.arange(50, dtype=np.float32).reshape(1, 50)
    orig = x.copy()
    smooth_tail_inplace(x, window_length=11)
    # window=11 -> tail=5. Positions 5..44 (interior) untouched.
    np.testing.assert_allclose(x[0, 5:45], orig[0, 5:45])


def test_shape_preserved():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(10, 100)).astype(np.float32)
    out = x.copy()
    smooth_tail_inplace(out, window_length=21)
    assert out.shape == (10, 100)


def test_ones_input_stays_ones():
    """Constant input should remain constant after tail smoothing."""
    x = np.ones((2, 200), dtype=np.float32)
    out = x.copy()
    smooth_tail_inplace(out, window_length=11)
    np.testing.assert_allclose(out, np.ones_like(x), atol=1e-5)


def test_invalid_window_returns_silently():
    x = np.zeros((1, 10), dtype=np.float32)
    smooth_tail_inplace(x, window_length=10)
    smooth_tail_inplace(x, window_length=2)

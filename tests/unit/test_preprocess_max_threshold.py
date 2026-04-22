"""Tests for pyinfercnv.preprocess.max_threshold."""
from __future__ import annotations

import numpy as np
import pytest

from pyinfercnv.preprocess.max_threshold import apply_max_centered_threshold


def test_clips_to_symmetric_bounds():
    X = np.array([[-5.0, -2.0, 0.0, 2.0, 5.0]], dtype=np.float32)
    out = apply_max_centered_threshold(X, threshold=3.0)
    np.testing.assert_allclose(out.ravel(), [-3.0, -2.0, 0.0, 2.0, 3.0])


def test_threshold_none_passes_through():
    X = np.array([[-100.0, 100.0]], dtype=np.float32)
    out = apply_max_centered_threshold(X, threshold=None)
    np.testing.assert_allclose(out, X)


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        apply_max_centered_threshold(np.zeros((1, 1), dtype=np.float32), threshold=-1.0)


def test_auto_not_implemented():
    with pytest.raises(NotImplementedError):
        apply_max_centered_threshold(np.zeros((1, 1), dtype=np.float32), threshold="auto")


def test_dtype_is_floating_point():
    """Phase 1 bit-exact path: leaf function returns float64; pipeline
    casts to float32 at result assembly only."""
    X = np.zeros((3, 3), dtype=np.float64)
    out = apply_max_centered_threshold(X, threshold=1.0)
    assert np.issubdtype(out.dtype, np.floating)

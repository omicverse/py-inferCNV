"""Tests for pyinfercnv.center.center_cells."""
from __future__ import annotations

import numpy as np
import pytest

from pyinfercnv.center.center_cells import center_cells


def test_median_subtract():
    X = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=np.float32)
    out = center_cells(X, method="median")
    np.testing.assert_allclose(out, [[-2, -1, 0, 1, 2], [-20, -10, 0, 10, 20]], atol=1e-6)


def test_mean_subtract():
    X = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    out = center_cells(X, method="mean")
    np.testing.assert_allclose(out, [[-2, -1, 0, 1, 2]], atol=1e-6)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        center_cells(np.zeros((1, 3), dtype=np.float32), method="mode")


def test_dtype_is_floating_point():
    """Phase 1 bit-exact path (2026-04-23): leaf returns float64; public
    float32 contract enforced at result.cnv_matrix in pipeline.py."""
    X = np.zeros((3, 3), dtype=np.float64)
    out = center_cells(X, method="median")
    assert np.issubdtype(out.dtype, np.floating)

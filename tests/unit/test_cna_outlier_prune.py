"""Tests for pyinfercnv.cna.outlier_prune."""
from __future__ import annotations

import numpy as np

from pyinfercnv.cna.outlier_prune import prune_outliers


def test_explicit_bounds_clip():
    X = np.array([[-5, -2, 0, 2, 5]], dtype=np.float32)
    out = prune_outliers(X, lower_bound=-3.0, upper_bound=3.0)
    np.testing.assert_allclose(out, [[-3, -2, 0, 2, 3]])


def test_average_bound_method_asymmetric_means():
    """R-parity: lower = mean(per-cell min), upper = mean(per-cell max)."""
    X = np.array([[-4, 0, 4], [-6, 0, 6]], dtype=np.float32)
    # per-cell min = [-4, -6], mean = -5; per-cell max = [4, 6], mean = 5
    out = prune_outliers(X, method="average_bound")
    assert out.max() <= 5.0 + 1e-6
    assert out.min() >= -5.0 - 1e-6


def test_average_bound_asymmetric_data():
    """Asymmetric: per-cell min mean=2, per-cell max mean=10 -> bounds [2, 10]."""
    X = np.array([[2, 5, 10], [2, 5, 10]], dtype=np.float32)
    out = prune_outliers(X, method="average_bound")
    # values 2, 5, 10 are within [2, 10] → unchanged
    np.testing.assert_allclose(out, X)


def test_none_returns_copy():
    X = np.array([[-100.0, 100.0]], dtype=np.float32)
    out = prune_outliers(X, method=None, lower_bound=None, upper_bound=None)
    np.testing.assert_allclose(out, X)


def test_unknown_method_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown outlier method"):
        prune_outliers(np.zeros((1, 1), dtype=np.float32), method="bogus")

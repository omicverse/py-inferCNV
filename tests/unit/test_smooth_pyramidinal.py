"""Tests for pyinfercnv.smooth.pyramidinal."""
from __future__ import annotations

import numpy as np

from pyinfercnv.smooth.pyramidinal import smooth_pyramidinal


def test_constant_row_is_unchanged():
    X = np.full((3, 200), 2.0, dtype=np.float32)
    out = smooth_pyramidinal(X, window_length=11)
    np.testing.assert_allclose(out, X, atol=1e-5)


def test_smoothing_reduces_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(2, 200)).astype(np.float32)
    out = smooth_pyramidinal(X, window_length=11)
    interior_var_out = out[:, 30:170].var(axis=1).mean()
    interior_var_in = X[:, 30:170].var(axis=1).mean()
    assert interior_var_out < interior_var_in * 0.6


def test_too_short_smoothes_with_tail_only():
    """If n_genes < window_length, only tail-helper runs (interior would be empty)."""
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = smooth_pyramidinal(X, window_length=11)
    assert out.shape == (1, 3)


def test_output_shape_preserved():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(5, 100)).astype(np.float32)
    out = smooth_pyramidinal(X, window_length=21)
    assert out.shape == X.shape


def test_invalid_window_raises():
    import pytest
    with pytest.raises(ValueError, match="window_length"):
        smooth_pyramidinal(np.zeros((1, 10), dtype=np.float32), window_length=10)

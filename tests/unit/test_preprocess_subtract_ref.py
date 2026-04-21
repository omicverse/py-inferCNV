"""Tests for pyinfercnv.preprocess.subtract_ref."""
from __future__ import annotations

import numpy as np

from pyinfercnv.preprocess.subtract_ref import subtract_reference


def test_single_reference_group_mean_path():
    X = np.array([[1, 2, 3], [3, 4, 5], [10, 10, 10], [0, 0, 0]], dtype=np.float32)
    out = subtract_reference(X, ref_groups={"normal": [0, 1]}, use_bounds=False)
    expected = np.array([[-1, -1, -1], [1, 1, 1], [8, 7, 6], [-2, -3, -4]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_multi_group_bounded_path_between_bounds_becomes_zero():
    X = np.array([[5.0], [10.0], [7.0], [12.0], [3.0]], dtype=np.float32)
    out = subtract_reference(X, ref_groups={"A": [0], "B": [1]}, use_bounds=True)
    np.testing.assert_allclose(out.ravel(), [0, 0, 0, 2, -2], atol=1e-6)


def test_bounded_path_single_group_equivalent_to_mean_path():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    out_mean = subtract_reference(X, ref_groups={"r": [0, 1]}, use_bounds=False)
    out_bnd = subtract_reference(X, ref_groups={"r": [0, 1]}, use_bounds=True)
    np.testing.assert_allclose(out_mean, out_bnd, atol=1e-6)


def test_output_dtype_float32():
    X = np.ones((3, 3), dtype=np.float32)
    out = subtract_reference(X, ref_groups={"r": [0]}, use_bounds=False)
    assert out.dtype == np.float32


def test_proxy_normal_fallback_when_no_ref():
    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    out = subtract_reference(X, ref_groups={"proxyNormal": [0, 1]}, use_bounds=False)
    np.testing.assert_allclose(out, [[-1, -1], [1, 1]], atol=1e-6)

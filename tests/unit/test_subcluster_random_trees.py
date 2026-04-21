"""Tests for pyinfercnv.subcluster.random_trees (G1 patch P7 applied)."""
from __future__ import annotations

import numpy as np
import pytest

from pyinfercnv.subcluster import random_tree_subcluster


def _synthetic_cnv_matrix(n_cells: int, n_genes: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    a = rng.normal(loc=-0.4, scale=0.05, size=(half, n_genes)).astype(np.float32)
    b = rng.normal(loc=+0.4, scale=0.05, size=(n_cells - half, n_genes)).astype(np.float32)
    return np.vstack([a, b])


def test_shape_and_dtype_small():
    X = _synthetic_cnv_matrix(50, n_genes=80)
    labels = random_tree_subcluster(
        X,
        subsample_for_tree=500,
        tumor_subcluster_pval=0.1,
        n_rand_iters=10,
        random_state=0,
    )
    assert labels.shape == (50,)
    assert labels.dtype == np.int32


def test_deterministic_with_fixed_seed():
    X = _synthetic_cnv_matrix(60, n_genes=80)
    a = random_tree_subcluster(
        X, subsample_for_tree=500, tumor_subcluster_pval=0.1,
        n_rand_iters=10, random_state=7,
    )
    b = random_tree_subcluster(
        X, subsample_for_tree=500, tumor_subcluster_pval=0.1,
        n_rand_iters=10, random_state=7,
    )
    np.testing.assert_array_equal(a, b)


def test_p7_subsample_warns_and_assigns_all_cells():
    """n_cells=700 > subsample_for_tree=500 → UserWarning + shape (700,).

    Uses the R-default permutation count of 100 to exercise the full null
    distribution path (runs in <2s on a 500x60 subsample).
    """
    X = _synthetic_cnv_matrix(700, n_genes=60)
    with pytest.warns(UserWarning, match="subsample_for_tree"):
        labels = random_tree_subcluster(
            X,
            subsample_for_tree=500,
            tumor_subcluster_pval=0.1,
            n_rand_iters=100,
            random_state=0,
        )
    assert labels.shape == (700,)
    assert labels.dtype == np.int32
    # every cell assigned a positive label
    assert labels.min() >= 1


def test_p7_no_warning_when_under_cap():
    import warnings as _warnings

    X = _synthetic_cnv_matrix(100, n_genes=60)
    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        _ = random_tree_subcluster(
            X,
            subsample_for_tree=500,
            tumor_subcluster_pval=0.1,
            n_rand_iters=5,
            random_state=0,
        )
    subsample_warnings = [
        w for w in record
        if issubclass(w.category, UserWarning) and "subsample_for_tree" in str(w.message)
    ]
    assert subsample_warnings == []


def test_labels_are_one_based():
    X = _synthetic_cnv_matrix(40, n_genes=60)
    labels = random_tree_subcluster(
        X, subsample_for_tree=500, tumor_subcluster_pval=0.1,
        n_rand_iters=5, random_state=0,
    )
    assert labels.min() >= 1


def test_too_few_cells_returns_ones():
    X = np.zeros((2, 10), dtype=np.float32)
    labels = random_tree_subcluster(X, subsample_for_tree=500, random_state=0)
    assert labels.shape == (2,)
    assert np.all(labels == 1)


def test_empty_input_returns_empty():
    X = np.zeros((0, 10), dtype=np.float32)
    labels = random_tree_subcluster(X, subsample_for_tree=500, random_state=0)
    assert labels.shape == (0,)
    assert labels.dtype == np.int32


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="2D"):
        random_tree_subcluster(np.zeros(10, dtype=np.float32), random_state=0)

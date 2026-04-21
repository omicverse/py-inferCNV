"""Tests for pyinfercnv.subcluster.qnorm."""
from __future__ import annotations

import numpy as np
import pytest

from pyinfercnv.subcluster import qnorm_subcluster


def _synthetic_cnv_matrix(n_cells: int, n_genes: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    a = rng.normal(loc=-0.5, scale=0.05, size=(half, n_genes)).astype(np.float32)
    b = rng.normal(loc=+0.5, scale=0.05, size=(n_cells - half, n_genes)).astype(np.float32)
    return np.vstack([a, b])


def test_shape_and_dtype():
    X = _synthetic_cnv_matrix(60)
    labels = qnorm_subcluster(X, tumor_subcluster_pval=0.1)
    assert labels.shape == (60,)
    assert labels.dtype == np.int32


def test_deterministic_repeat_runs_identical():
    X = _synthetic_cnv_matrix(60)
    a = qnorm_subcluster(X, tumor_subcluster_pval=0.1)
    b = qnorm_subcluster(X, tumor_subcluster_pval=0.1)
    np.testing.assert_array_equal(a, b)


def test_labels_are_one_based():
    X = _synthetic_cnv_matrix(40)
    labels = qnorm_subcluster(X, tumor_subcluster_pval=0.1)
    assert labels.min() >= 1


def test_small_input_returns_single_cluster():
    X = np.random.default_rng(0).normal(size=(2, 10)).astype(np.float32)
    labels = qnorm_subcluster(X, tumor_subcluster_pval=0.1)
    assert np.all(labels == 1)


def test_two_clear_blocks_produce_at_least_two_clusters():
    """With a strong 2-block signal and p_val=0.1, qnorm should typically
    yield ≥ 2 clusters. This is not a bit-exact R comparison but a sanity
    floor."""
    X = _synthetic_cnv_matrix(80, n_genes=120, seed=1)
    labels = qnorm_subcluster(X, tumor_subcluster_pval=0.1)
    assert len(np.unique(labels)) >= 2


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="2D"):
        qnorm_subcluster(np.zeros(10, dtype=np.float32))

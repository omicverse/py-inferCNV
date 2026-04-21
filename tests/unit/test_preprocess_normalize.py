"""Tests for pyinfercnv.preprocess.normalize."""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from pyinfercnv.preprocess.normalize import normalize_by_seq_depth


def test_median_libsize_normalization_dense():
    X = np.array([[5, 5], [10, 10], [15, 15]], dtype=np.float32)
    out = normalize_by_seq_depth(X)
    np.testing.assert_allclose(
        out if not sp.issparse(out) else out.toarray(),
        [[10, 10], [10, 10], [10, 10]],
    )


def test_sparse_input_preserved_sparse():
    X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
    out = normalize_by_seq_depth(X)
    assert sp.issparse(out)


def test_custom_normalize_factor():
    X = np.array([[1, 1], [2, 2]], dtype=np.float32)
    out = normalize_by_seq_depth(X, normalize_factor=100.0)
    np.testing.assert_allclose(
        out if not sp.issparse(out) else out.toarray(), [[50, 50], [50, 50]]
    )


def test_zero_libsize_cell_becomes_zero():
    X = np.array([[0, 0], [2, 2]], dtype=np.float32)
    out = normalize_by_seq_depth(X)
    dense = out if not sp.issparse(out) else out.toarray()
    assert (dense[0] == 0).all()

"""Tests for pyinfercnv.preprocess.log_transform."""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from pyinfercnv.preprocess.log_transform import invert_log2, invert_log2_plus1, log2_plus1


def test_log2_plus1_dense():
    x = np.array([0.0, 1.0, 3.0, 7.0], dtype=np.float32)
    out = log2_plus1(x)
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0], atol=1e-6)


def test_log2_plus1_sparse_densifies_correctly():
    X = sp.csr_matrix(np.array([[0, 1, 3], [0, 7, 0]], dtype=np.float32))
    out = log2_plus1(X)
    assert sp.issparse(out)
    expected = np.log2(np.array([[0, 1, 3], [0, 7, 0]]) + 1)
    np.testing.assert_allclose(out.toarray(), expected, atol=1e-6)


def test_invert_round_trip():
    x = np.array([0.0, 0.5, 1.0, 2.5], dtype=np.float32)
    round_trip = invert_log2_plus1(log2_plus1(x))
    np.testing.assert_allclose(round_trip, x, atol=1e-6)


def test_invert_log2_is_2_to_x():
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(invert_log2(x), [0.5, 1.0, 2.0, 4.0], atol=1e-6)

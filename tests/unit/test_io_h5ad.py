"""Tests for pyinfercnv.io.h5ad.extract_counts."""
from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse as sp

from pyinfercnv.io.h5ad import extract_counts


def _build_adata(X, has_counts_layer=True):
    X_for_X = X.astype(np.float32) if not sp.issparse(X) else X.astype(np.float32)
    ad = AnnData(X=X_for_X)
    if has_counts_layer:
        ad.layers["counts"] = X.copy() if hasattr(X, "copy") else X
    return ad


def test_default_reads_counts_layer():
    X = sp.csr_matrix(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ad = _build_adata(X, has_counts_layer=True)
    out = extract_counts(ad)
    assert sp.issparse(out)
    assert np.array_equal(out.toarray(), [[1, 2, 3], [4, 5, 6]])


def test_missing_counts_layer_raises():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ad = _build_adata(X, has_counts_layer=False)
    with pytest.raises(KeyError, match="layers.*counts"):
        extract_counts(ad, counts_layer="counts")


def test_counts_layer_none_integer_x_ok():
    X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.int32))
    ad = _build_adata(X, has_counts_layer=False)
    out = extract_counts(ad, counts_layer=None)
    assert np.array_equal(out.toarray(), [[1, 2], [3, 4]])


def test_counts_layer_none_float_non_integer_warns():
    X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    ad = _build_adata(X, has_counts_layer=False)
    with pytest.warns(UserWarning, match="not look like counts"):
        extract_counts(ad, counts_layer=None)


def test_counts_layer_none_float_integer_valued_ok():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ad = _build_adata(X, has_counts_layer=False)
    out = extract_counts(ad, counts_layer=None)
    assert np.array_equal(out.toarray() if sp.issparse(out) else out, [[1, 2], [3, 4]])


def test_custom_layer_name():
    X = sp.csr_matrix(np.array([[2, 3]], dtype=np.int32))
    ad = AnnData(X=np.zeros((1, 2), dtype=np.float32))
    ad.layers["raw"] = X
    out = extract_counts(ad, counts_layer="raw")
    assert np.array_equal(out.toarray(), [[2, 3]])


def test_output_is_csr_float32():
    X = sp.csc_matrix(np.array([[1, 2], [3, 4]], dtype=np.int32))
    ad = _build_adata(X, has_counts_layer=True)
    out = extract_counts(ad)
    assert sp.isspmatrix_csr(out)
    assert out.dtype == np.float32

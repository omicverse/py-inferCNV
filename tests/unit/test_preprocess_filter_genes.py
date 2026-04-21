"""Tests for pyinfercnv.preprocess.filter_genes."""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from pyinfercnv.preprocess.filter_genes import filter_low_expression_genes


def test_removes_genes_below_mean_cutoff():
    X = sp.csr_matrix(np.array([[0, 5, 0], [0, 5, 0], [0, 5, 0]], dtype=np.float32))
    mask = filter_low_expression_genes(X, cutoff=1.0, min_cells_per_gene=0)
    assert mask.tolist() == [False, True, False]


def test_requires_min_cells():
    X = sp.csr_matrix(np.array([[10, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32))
    mask = filter_low_expression_genes(X, cutoff=0.0, min_cells_per_gene=2)
    assert mask.tolist() == [False, False, True]


def test_reference_only_filter_respects_ref_indices():
    X = sp.csr_matrix(np.array([[10, 0], [10, 0], [0, 5], [0, 5]], dtype=np.float32))
    ref_idx = [0, 1]
    mask = filter_low_expression_genes(
        X, cutoff=1.0, min_cells_per_gene=1, reference_cell_idx=ref_idx
    )
    assert mask.tolist() == [True, False]


def test_dense_and_sparse_equivalence():
    rng = np.random.default_rng(0)
    dense = rng.poisson(0.5, size=(10, 20)).astype(np.float32)
    sparse_X = sp.csr_matrix(dense)
    m1 = filter_low_expression_genes(dense, cutoff=0.1, min_cells_per_gene=2)
    m2 = filter_low_expression_genes(sparse_X, cutoff=0.1, min_cells_per_gene=2)
    assert np.array_equal(m1, m2)

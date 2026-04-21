"""CPM-by-median-libsize normalization, R-parity with `normalize_counts_by_seq_depth`
(`inferCNV_ops.R:3082-3110`).

factor = median(colSums in R semantics, == row_sums in cells x genes layout);
each cell = cell / cell_libsize * factor.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp


def normalize_by_seq_depth(X, *, normalize_factor: float | None = None):
    """Scale each cell (row) so row-sum equals `normalize_factor`.

    If `normalize_factor` is None, use median(row_sums) (R default).
    Zero-libsize cells are left as zero (no division by zero).
    """
    if sp.issparse(X):
        Xc = X.tocsr().astype(np.float32, copy=False)
        libsizes = np.asarray(Xc.sum(axis=1)).ravel().astype(np.float64)
    else:
        Xc = np.asarray(X, dtype=np.float32)
        libsizes = Xc.sum(axis=1).astype(np.float64)

    if normalize_factor is None:
        normalize_factor = float(np.median(libsizes))

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(libsizes > 0, normalize_factor / libsizes, 0.0)

    if sp.issparse(Xc):
        diag = sp.diags(scale.astype(np.float32))
        return (diag @ Xc).tocsr()
    return (Xc * scale[:, None]).astype(np.float32)

"""CPM-by-median-libsize normalization, R-parity with `normalize_counts_by_seq_depth`
(`inferCNV_ops.R:3082-3110`).

factor = median(colSums in R semantics, == row_sums in cells x genes layout);
each cell = cell / cell_libsize * factor.

Phase 1 bit-exact path (2026-04-23): intermediate computation is float64
throughout. The pipeline casts down to float32 only at the final result
assembly (`result.cnv_matrix`); callers of this function get float64 dense
arrays back when the input is dense float64, preserving R's numerical
precision through the rest of Phase 1.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp


def normalize_by_seq_depth(X, *, normalize_factor: float | None = None):
    """Scale each cell (row) so row-sum equals `normalize_factor`.

    If `normalize_factor` is None, use median(row_sums) (R default).
    Zero-libsize cells are left as zero (no division by zero).

    Dtype policy (bit-exact path):
      * sparse input  -> sparse float64 output
      * dense input   -> dense float64 output
    The pipeline downcasts at result assembly; callers wanting float32 can
    `.astype(np.float32)` on return.
    """
    if sp.issparse(X):
        Xc = X.tocsr().astype(np.float64, copy=False)
        libsizes = np.asarray(Xc.sum(axis=1)).ravel()
    else:
        Xc = np.asarray(X, dtype=np.float64)
        libsizes = Xc.sum(axis=1)

    if normalize_factor is None:
        normalize_factor = float(np.median(libsizes))

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(libsizes > 0, normalize_factor / libsizes, 0.0)

    if sp.issparse(Xc):
        diag = sp.diags(scale)
        return (diag @ Xc).tocsr()
    return Xc * scale[:, None]

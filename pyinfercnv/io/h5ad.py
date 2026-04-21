"""AnnData reader + counts_layer sniffer.

Contract (spec section 6.1):
    - Default `counts_layer="counts"`: read adata.layers["counts"]; raise if missing.
    - `counts_layer=None`: use adata.X; warn if it doesn't look like counts.
    - `counts_layer="<other>"`: read named layer.
    - Always return CSR float32.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse as sp

if TYPE_CHECKING:
    from anndata import AnnData


def _looks_like_counts(x) -> bool:
    """Heuristic — integer dtype OR float-but-all-integer-values."""
    if sp.issparse(x):
        if np.issubdtype(x.dtype, np.integer):
            return True
        data = x.data
    else:
        arr = np.asarray(x)
        if np.issubdtype(arr.dtype, np.integer):
            return True
        data = arr.ravel()
    if data.size == 0:
        return True
    sample = data if data.size <= 100_000 else data[:: max(1, data.size // 100_000)]
    return bool(np.all(sample == sample.astype(np.int64).astype(sample.dtype)))


def _to_csr_float32(x) -> sp.csr_matrix:
    if sp.issparse(x):
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))


def extract_counts(adata: "AnnData", *, counts_layer: str | None = "counts") -> sp.csr_matrix:
    """Return CSR float32 counts matrix (cells x genes) per contract."""
    if counts_layer is None:
        X = adata.X
        if not _looks_like_counts(X):
            warnings.warn(
                "adata.X does not look like counts (non-integer float values detected); "
                "CPM normalization may produce incorrect results. Pass counts_layer='<name>' "
                "to point at a raw counts layer, or rerun upstream normalization.",
                UserWarning,
                stacklevel=2,
            )
        return _to_csr_float32(X)

    if counts_layer not in adata.layers:
        raise KeyError(
            f"adata.layers[{counts_layer!r}] not found. "
            f"Run ov.pp.qc(adata) first, or pass counts_layer=None to use adata.X."
        )
    return _to_csr_float32(adata.layers[counts_layer])

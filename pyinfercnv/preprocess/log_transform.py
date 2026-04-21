"""log2(x+1) and inverses — R parity with `log2xplus1` / `invert_log2xplus1` /
`invert_log2` (`inferCNV_ops.R:2756-2826`).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp


def log2_plus1(X):
    """log2(X + 1). Sparse-preserving via log1p."""
    if sp.issparse(X):
        out = X.copy().astype(np.float32)
        out.data = (np.log1p(out.data) / np.log(2.0)).astype(np.float32)
        return out
    return (np.log1p(np.asarray(X, dtype=np.float32)) / np.log(2.0)).astype(np.float32)


def invert_log2_plus1(X):
    """2**X - 1. Inverse of log2_plus1."""
    if sp.issparse(X):
        out = X.copy().astype(np.float32)
        out.data = (np.exp2(out.data) - 1.0).astype(np.float32)
        return out
    return (np.exp2(np.asarray(X, dtype=np.float32)) - 1.0).astype(np.float32)


def invert_log2(X):
    """2**X. Used at R step 14 to move log2(FC) -> linear FC space."""
    if sp.issparse(X):
        return np.exp2(X.toarray().astype(np.float32)).astype(np.float32)
    return np.exp2(np.asarray(X, dtype=np.float32)).astype(np.float32)

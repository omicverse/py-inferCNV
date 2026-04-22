"""Per-cell centering across genes — R parity with `center_cell_expr_across_chromosome`
(`inferCNV_ops.R:2074-2109`).
"""
from __future__ import annotations

import numpy as np


def center_cells(X: np.ndarray, *, method: str = "median") -> np.ndarray:
    Xc = np.asarray(X, dtype=np.float64)
    if method == "median":
        c = np.median(Xc, axis=1, keepdims=True)
    elif method == "mean":
        c = np.mean(Xc, axis=1, keepdims=True)
    else:
        raise ValueError(f"method must be 'median' or 'mean', got {method!r}")
    return Xc - c

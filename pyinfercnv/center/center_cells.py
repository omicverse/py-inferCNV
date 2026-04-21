"""Per-cell centering across genes — R parity with `center_cell_expr_across_chromosome`
(`inferCNV_ops.R:2074-2109`).
"""
from __future__ import annotations

import numpy as np


def center_cells(X: np.ndarray, *, method: str = "median") -> np.ndarray:
    if method == "median":
        c = np.median(X, axis=1, keepdims=True)
    elif method == "mean":
        c = np.mean(X, axis=1, keepdims=True)
    else:
        raise ValueError(f"method must be 'median' or 'mean', got {method!r}")
    return (X - c).astype(np.float32)

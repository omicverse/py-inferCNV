"""Pyramidinal (triangular-kernel) smoothing across genes.

R parity:
    interior = .smooth_center_helper(obs, window_length)
        kernel = [1, 2, ..., tail, tail+1, tail, ..., 2, 1] / (tail^2 + window_length)
        Implemented bit-exact in float64 via a single centered direct
        convolution with pre-divided coefficients, matching R's
        `stats::filter(vals, custom_filter, sides=2)` (see
        `pyinfercnv.kernels.smooth_center_numba.smooth_center_interior`).

    tail = pyinfercnv.kernels.smooth_tail_numba.smooth_tail_overwrite
        Replaces positions [0..tail-1] and [n-tail..n-1] with R's dynamic-denominator
        tail formula (see kernel docstring).

R source: smooth_by_chromosome / .smooth_window / .smooth_helper / .smooth_center_helper
(`inferCNV_ops.R:2440-2660`).
"""
from __future__ import annotations

import numpy as np

from pyinfercnv.kernels.smooth_center_numba import smooth_center_interior
from pyinfercnv.kernels.smooth_tail_numba import smooth_tail_overwrite


def smooth_pyramidinal(X: np.ndarray, *, window_length: int = 101) -> np.ndarray:
    """Smooth each row of X by R-parity triangular kernel of size `window_length`.

    X has shape (n_cells, n_genes_on_chromosome). Returned shape == input shape.
    """
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(f"window_length must be odd and >= 3, got {window_length}")
    # Phase 1 bit-exact path (2026-04-23): float64 throughout.
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    n_cells, n_genes = X64.shape

    if n_genes < 2:
        return X64.copy()

    if n_genes < window_length:
        # R's tail-only behaviour: smooth_helper iterates ceil(n/2) positions
        # and overwrites both ends using the ORIGINAL data (no center pass).
        out = X64.copy()
        smooth_tail_overwrite(out, X64, window_length)
        return out

    # R-exact interior: single centered direct convolution with pre-divided
    # triangular kernel, float64 accumulation (bit-exact with R stats::filter).
    out = X64.copy()
    smooth_center_interior(out, X64, window_length)

    # R-exact tail: values drawn from the ORIGINAL input X64, not from the
    # interior-smoothed `out`. Overwrites positions [0..tail-1] and [n-tail..n-1].
    smooth_tail_overwrite(out, X64, window_length)
    return out

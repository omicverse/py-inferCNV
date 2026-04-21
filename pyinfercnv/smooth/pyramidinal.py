"""Pyramidinal (triangular-kernel) smoothing across genes.

R parity:
    interior = .smooth_center_helper(obs, window_length)
        kernel = [1, 2, ..., tail, tail+1, tail, ..., 2, 1] / (tail^2 + window_length)
        Implemented bit-exact via two scipy.ndimage.uniform_filter1d passes of
        size = (window_length + 1) // 2. Mathematical identity:
            box(half_w) * box(half_w) = triangle of width 2*half_w-1 = window_length
        Sum of triangle weights = half_w^2 = ((window_length+1)/2)^2 = tail^2 + window_length.

    tail = pyinfercnv.kernels.smooth_tail_numba.smooth_tail_inplace
        Replaces positions [0..tail-1] and [n-tail..n-1] with R's dynamic-denominator
        tail formula (see kernel docstring).

R source: smooth_by_chromosome / .smooth_window / .smooth_helper / .smooth_center_helper
(`inferCNV_ops.R:2440-2660`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from pyinfercnv.kernels.smooth_tail_numba import smooth_tail_overwrite


def smooth_pyramidinal(X: np.ndarray, *, window_length: int = 101) -> np.ndarray:
    """Smooth each row of X by R-parity triangular kernel of size `window_length`.

    X has shape (n_cells, n_genes_on_chromosome). Returned shape == input shape.
    """
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(f"window_length must be odd and >= 3, got {window_length}")
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    n_cells, n_genes = X32.shape

    if n_genes < 2:
        return X32.copy()

    if n_genes < window_length:
        # R's tail-only behaviour: smooth_helper iterates ceil(n/2) positions
        # and overwrites both ends using the ORIGINAL data (no center pass).
        out = X32.copy()
        smooth_tail_overwrite(out, X32, window_length)
        return out

    half_w = (window_length + 1) // 2
    out = ndimage.uniform_filter1d(X32, size=half_w, axis=1, mode="nearest")
    out = ndimage.uniform_filter1d(out, size=half_w, axis=1, mode="nearest")
    out = np.ascontiguousarray(out, dtype=np.float32)

    # R-exact tail: values drawn from the ORIGINAL input X32, not from the
    # pre-smoothed `out`. Overwrites positions [0..tail-1] and [n-tail..n-1].
    smooth_tail_overwrite(out, X32, window_length)
    return out

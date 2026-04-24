"""Numba @njit implementation of R `.smooth_center_helper` interior convolution.

R source (inferCNV_ops.R:2640-2661):

    custom_filter_denominator = ((window_length-1)/2)^2 + window_length
    custom_filter_numerator   = c(seq_len(tail), tail+1, c(tail:1))
    custom_filter = custom_filter_numerator / custom_filter_denominator
    smoothed = stats::filter(vals, custom_filter, sides=2)

`stats::filter(..., sides=2)` performs a single centered direct convolution
with pre-divided float64 coefficients. At each interior position t
(tail <= t <= n - tail - 1) it computes, left-to-right:

    out[t] = sum_{k=0..window_length-1} x[t - tail + k] * coef[k]

where `coef[k] = numer[k] / denom` is precomputed. This is bit-exact with
R's `do_cfilter` (src/library/stats/src/filter.c).

Boundary positions [0..tail-1] and [n-tail..n-1] are left untouched here;
R's `stats::filter` returns NA for them and the caller overwrites via the
dynamic-denominator tail formula (see smooth_tail_numba.py).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False)
def smooth_center_interior(out: np.ndarray, obs: np.ndarray, window_length: int) -> None:
    """Fill interior positions of `out` with R-exact triangular-kernel convolution of `obs`.

    Parameters
    ----------
    out : (n_rows, n_genes) float64, written in place at interior positions.
    obs : (n_rows, n_genes) float64, read-only source.
    window_length : odd int >= 3.

    Only positions [tail..n_genes-tail-1] are written. Boundary positions are
    NOT touched. Caller is responsible for seeding `out` (e.g. `out = obs.copy()`)
    if it later overwrites the tail via `smooth_tail_overwrite`.
    """
    if window_length < 3 or window_length % 2 == 0:
        return
    n_rows, n_genes = obs.shape
    if n_genes < window_length:
        return

    tail = (window_length - 1) // 2

    # R: custom_filter = numer / denom, computed ONCE in float64 before the
    # convolution. Matches do_cfilter's pre-divided coefficient semantics.
    denom = float(tail * tail + window_length)
    coef = np.empty(window_length, dtype=np.float64)
    for i in range(tail):
        coef[i] = float(i + 1) / denom
    coef[tail] = float(tail + 1) / denom
    for i in range(tail):
        coef[tail + 1 + i] = float(tail - i) / denom

    last_interior = n_genes - tail  # exclusive

    for r in prange(n_rows):
        for t in range(tail, last_interior):
            base = t - tail
            s = 0.0
            for k in range(window_length):
                s += obs[r, base + k] * coef[k]
            out[r, t] = s

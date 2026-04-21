"""Numba @njit implementation of R `.smooth_helper` tail-weighting (inferCNV_ops.R:2483-2532).

Direct R port. R algorithm for tail smoothing is applied AFTER the interior
triangular-kernel convolution (`.smooth_center_helper`):

    end_data <- .smooth_center_helper(obs_data, window_length)  # interior positions
    # tail positions overwritten here — using ORIGINAL obs_data, not end_data:
    for tail_end in 1..iteration_range:
        end_tail = obs_count - tail_end + 1
        d_left  = tail_end - 1
        d_right = min(obs_count - tail_end, tail_length)
        r_left  = tail_length - d_left
        r_right = tail_length - d_right
        denom   = tail_length^2 + window_length - r_left*(r_left+1)/2 - r_right*(r_right+1)/2
        end_data[tail_end] = sum(obs_data[1:(tail_end+d_right)] * numer[(tail+1-d_left):(tail+1+d_right)]) / denom
        end_data[end_tail] = sum(obs_data[(end_tail-d_right):obs_length] * rev(numer[...])) / denom

Critical detail (previously missed): the tail uses **original input** for the
weighted sum, not the pre-smoothed data. We therefore take two matrices:
    `out`    — interior-smoothed (will be overwritten at tail positions)
    `obs`    — original input (source of values for tail weighted sums)
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False)
def smooth_tail_overwrite(out: np.ndarray, obs: np.ndarray, window_length: int) -> None:
    """Overwrite tail positions of `out` using R-exact tail formula on `obs`.

    `out`    (n_cells, n_genes)  float32  — will be modified in place
    `obs`    (n_cells, n_genes)  float32  — original input, read-only source

    Positions [0..tail-1] and [n-tail..n-1] of each row of `out` are replaced.
    Interior positions are untouched.
    """
    if window_length < 3 or window_length % 2 == 0:
        return
    n_rows, n_genes = out.shape
    if n_genes < 2:
        return

    tail = (window_length - 1) // 2

    if n_genes > window_length:
        iteration_range = tail
    else:
        iteration_range = (n_genes + 1) // 2

    numer = np.empty(window_length, dtype=np.float64)
    for i in range(tail):
        numer[i] = float(i + 1)
    numer[tail] = float(tail + 1)
    for i in range(tail):
        numer[tail + 1 + i] = float(tail - i)

    for r in prange(n_rows):
        for t in range(iteration_range):
            d_left = t
            d_right_raw = n_genes - 1 - t
            d_right = tail if d_right_raw > tail else d_right_raw

            r_left = tail - d_left
            r_right = tail - d_right
            denom = (tail * tail + window_length
                     - (r_left * (r_left + 1)) // 2
                     - (r_right * (r_right + 1)) // 2)

            chunk_len = t + 1 + d_right
            numer_start = tail - d_left

            s_left = 0.0
            for k in range(chunk_len):
                s_left += float(obs[r, k]) * numer[numer_start + k]

            end_idx = n_genes - 1 - t
            right_start = end_idx - d_right
            s_right = 0.0
            for k in range(chunk_len):
                s_right += float(obs[r, right_start + k]) * numer[numer_start + (chunk_len - 1 - k)]

            out[r, t] = np.float32(s_left / denom)
            out[r, end_idx] = np.float32(s_right / denom)


@njit(cache=True, parallel=True, fastmath=False)
def smooth_tail_inplace(data: np.ndarray, window_length: int) -> None:
    """Back-compat shim: apply tail smoothing with `data` serving as both `out`
    and `obs`. This matches the previous behaviour where the tail uses the
    pre-smoothed interior values (INCORRECT vs R). Kept only for smooth_tail
    unit tests that don't care about R-parity correctness.
    """
    smooth_tail_overwrite(data, data, window_length)

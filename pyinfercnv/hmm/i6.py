"""6-state HMM Viterbi decoder — mirrors R `predict_CNV_via_HMM_on_indiv_cells`.

R reference (inferCNV_HMM.R):
    - `.get_HMM(cnv_mean_sd, t)`                    lines 230-265
    - `predict_CNV_via_HMM_on_indiv_cells`          lines 284-325

State indexing
--------------
R uses 1-based state labels {1..6} corresponding to CNV levels
(0.01, 0.5, 1.0, 1.5, 2.0, 3.0). Python port uses 0-based int8 labels {0..5}
(state 0 -> cnv 0.01, state 2 -> neutral cnv 1.0, state 5 -> cnv 3.0).

Per-chromosome decoding
-----------------------
R loops `lapply(chrs, ...)` so each chromosome is an independent sequence;
DP is reset at every chromosome boundary. We mirror this using `chr_pos`
(ordered mapping `chr -> start_col_idx`). CNV state does NOT carry over
chromosome breaks.

Default state_mus / state_sigmas
--------------------------------
R reads these from `cnv_mean_sd` which is fit from the hspike simulated
matrix (see `get_spike_dists` -> cnv levels "cnv:0.01", "cnv:0.5", ...,
"cnv:3"). The hspike pipeline is NOT yet ported. The default here is a
proxy: `mus = log2([0.01, 0.5, 1.0, 1.5, 2.0, 3.0])` with floor on the
CN=0 state (log2(0.01) ≈ -6.64 yields an extreme Gaussian that is rarely
called in post-smoothed infercnv space, which is acceptable as a
conservative default but NOT bit-exact to R). Users who want R parity
must pass fitted `state_mus` / `state_sigmas` explicitly.
"""
from __future__ import annotations

import numpy as np

from pyinfercnv.kernels.hmm_viterbi_numba import viterbi_decode_numba


#: R CNV levels from inferCNV_HMM.R:244-256 (state order: CN=0, het-del, neutral, het-amp, dup, amp)
I6_CNV_LEVELS: np.ndarray = np.array([0.01, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64)

#: Default proxy means in log2 space (see module docstring)
I6_DEFAULT_MUS: np.ndarray = np.log2(I6_CNV_LEVELS).astype(np.float64)

#: Default shared sigma; R uses per-state sigmas fit from hspike. Acceptable proxy.
I6_DEFAULT_SIGMA: float = 0.5

N_STATES_I6: int = 6


def _build_transition_matrix(n_states: int, t: float) -> np.ndarray:
    """R .get_HMM transition: diag = 1 - (K-1)*t, off-diag = t. Lines 233-240."""
    if t <= 0.0 or t >= 1.0 / (n_states - 1):
        raise ValueError(
            f"transition_prob must be in (0, 1/(K-1)=1/{n_states - 1}), got {t}"
        )
    trans = np.full((n_states, n_states), t, dtype=np.float64)
    np.fill_diagonal(trans, 1.0 - (n_states - 1) * t)
    return trans


def _build_initial_distribution(n_states: int, neutral_idx: int, t: float) -> np.ndarray:
    """R .get_HMM delta: `c(t, t, 1-5t, t, t, t)`, line 242.

    Mass 1 - (K-1)*t at the neutral state, t on every other state.
    """
    delta = np.full(n_states, t, dtype=np.float64)
    delta[neutral_idx] = 1.0 - (n_states - 1) * t
    return delta


def predict_i6(  # noqa: N802 — R kwargs preserved below
    cnv_matrix: np.ndarray,
    chr_pos: dict[str, int],
    *,
    transition_prob: float = 1e-6,
    state_mus: np.ndarray | None = None,
    state_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Return int8 array (n_cells, n_bins) of Viterbi-decoded 6-state CNV labels.

    Parameters
    ----------
    cnv_matrix
        Smoothed / centered infercnv expression matrix, shape (n_cells, n_bins),
        in log2-fold-change space.
    chr_pos
        Ordered mapping `chromosome -> start column index` (keys ordered by
        column). Last block extends to n_bins. Defines independent Viterbi
        segments per chromosome (mirrors `lapply(chrs, ...)` in R
        inferCNV_HMM.R:299).
    transition_prob
        Off-diagonal transition `t`. Must be in (0, 1/5). Corresponds to R
        `t` in `.get_HMM` (line 230).
    state_mus
        Optional shape (6,) Gaussian means per state (CN=0..CN=3+). Defaults
        to `I6_DEFAULT_MUS` = log2([0.01, 0.5, 1.0, 1.5, 2.0, 3.0]).
    state_sigmas
        Optional shape (6,) Gaussian stddevs per state. Defaults to
        `I6_DEFAULT_SIGMA` on every state.

    Returns
    -------
    np.ndarray
        Int8 (n_cells, n_bins). States 0..5 mapping to CN=0, het-del, neutral,
        het-amp, dup, high-amp.
    """
    cnv_matrix = np.asarray(cnv_matrix)
    if cnv_matrix.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2-D, got shape {cnv_matrix.shape}")
    n_cells, n_bins = cnv_matrix.shape

    if state_mus is None:
        mus = I6_DEFAULT_MUS.copy()
    else:
        mus = np.asarray(state_mus, dtype=np.float64).ravel()
    if mus.shape != (N_STATES_I6,):
        raise ValueError(f"state_mus must be shape ({N_STATES_I6},), got {mus.shape}")

    if state_sigmas is None:
        sigmas = np.full(N_STATES_I6, I6_DEFAULT_SIGMA, dtype=np.float64)
    else:
        sigmas = np.asarray(state_sigmas, dtype=np.float64).ravel()
    if sigmas.shape != (N_STATES_I6,):
        raise ValueError(f"state_sigmas must be shape ({N_STATES_I6},), got {sigmas.shape}")
    if np.any(sigmas <= 0.0):
        raise ValueError("state_sigmas must all be positive")

    log_trans = np.log(_build_transition_matrix(N_STATES_I6, transition_prob))
    neutral_idx = 2  # CN=1.0 == R state index 3 (1-based) == python state 2
    log_delta = np.log(_build_initial_distribution(N_STATES_I6, neutral_idx, transition_prob))

    # chromosome block edges
    starts = list(chr_pos.values())
    if starts and starts[0] != 0:
        raise ValueError(f"chr_pos first start must be 0, got {starts[0]}")
    ends = starts[1:] + [n_bins]

    out = np.empty((n_cells, n_bins), dtype=np.int8)
    cast_obs = cnv_matrix.astype(np.float64, copy=False)
    for c in range(n_cells):
        for start, end in zip(starts, ends, strict=True):
            if end <= start:
                continue
            seg = np.ascontiguousarray(cast_obs[c, start:end])
            path = viterbi_decode_numba(seg, log_delta, log_trans, mus, sigmas)
            out[c, start:end] = path.astype(np.int8)
    return out


__all__ = [
    "predict_i6",
    "I6_CNV_LEVELS",
    "I6_DEFAULT_MUS",
    "I6_DEFAULT_SIGMA",
    "N_STATES_I6",
]

"""3-state HMM (DEL, neutral, AMP) — mirrors R `inferCNV_i3HMM.R`.

R reference:
    - `.i3HMM_get_sd_trend_by_num_cells_fit`    lines 17-80
    - `.i3HMM_get_HMM`                          lines 99-156
    - `i3HMM_predict_CNV_via_HMM_on_indiv_cells` lines 180-225
    - `determine_mean_delta_via_Z`              lines 435-445

State indexing
--------------
R returns 1-based state labels {1, 2, 3} (DEL, neutral, AMP). Python port
uses 0-based int8 {0, 1, 2} (0=DEL, 1=neutral, 2=AMP).

Default parameter estimation
----------------------------
`estimate_i3_state_params(cnv_matrix, reference_cell_idx, i3_p_val=0.05)`
mirrors R `.i3HMM_get_sd_trend_by_num_cells_fit` + `determine_mean_delta_via_Z`:
    mu = mean(ref_values)
    sigma = std(ref_values, ddof=1)           # R sd() uses N-1 denominator
    mean_delta = |qnorm(p=i3_p_val, mean=0, sd=sigma)| = sigma * |z_{p}|
    mus = (mu - mean_delta, mu, mu + mean_delta)
    sigmas = (sigma, sigma, sigma)            # shared variance (R line 143-145)

This is the Z-based variant; R has an alternative KS variant
(`get_HoneyBADGER_setGexpDev`, lines 469-493) but it's stochastic
(uses `rnorm` sampling) so we default to the deterministic Z form.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pyinfercnv.kernels.hmm_viterbi_numba import viterbi_decode_numba


N_STATES_I3: int = 3

#: Default shared sigma if no reference supplied (proxy — rarely used in pipeline).
I3_DEFAULT_SIGMA: float = 0.5
#: Default symmetric mean offset in log2 space (arbitrary proxy).
I3_DEFAULT_MEAN_DELTA: float = 0.5


def _build_transition_matrix(n_states: int, t: float) -> np.ndarray:
    """R .i3HMM_get_HMM transition: 1-5*t on diag, t off-diag (lines 108-112).

    Note: R uses `1-5*t` on the diagonal even for K=3 — this is a verbatim
    copy from the i6 form. It means rows do not sum to 1 for K<6; we
    preserve the R exact behaviour for audit parity but validate t-range
    against the strictest (K=6) bound `1/(K-1)=0.2`.
    """
    if t <= 0.0 or t >= 0.2:
        raise ValueError(f"transition_prob must be in (0, 0.2), got {t}")
    trans = np.full((n_states, n_states), t, dtype=np.float64)
    # R writes `1-5*t` on diagonal — we mirror this exactly (line 108-110)
    np.fill_diagonal(trans, 1.0 - 5.0 * t)
    return trans


def _build_initial_distribution(n_states: int, neutral_idx: int, t: float) -> np.ndarray:
    """R .i3HMM_get_HMM delta: `c(t, 1-5t, t)`, line 114."""
    delta = np.full(n_states, t, dtype=np.float64)
    delta[neutral_idx] = 1.0 - 5.0 * t
    return delta


def estimate_i3_state_params(
    cnv_matrix: np.ndarray,
    reference_cell_idx: np.ndarray | list[int],
    i3_p_val: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (mus, sigmas) for i3 HMM from reference-cell distribution.

    Mirrors R `.i3HMM_get_sd_trend_by_num_cells_fit` (inferCNV_i3HMM.R:17-80)
    + `determine_mean_delta_via_Z` (lines 435-445).

    Parameters
    ----------
    cnv_matrix
        Shape (n_cells, n_bins). In log2-space (centered).
    reference_cell_idx
        Integer indices or boolean mask of reference ("normal") cells.
    i3_p_val
        p-value controlling distance between neutral and DEL/AMP means
        (R kwarg `i3_p_val`, default 0.05).

    Returns
    -------
    mus
        Shape (3,): (mu - delta, mu, mu + delta).
    sigmas
        Shape (3,): (sigma, sigma, sigma) — shared variance per R line 143-145.
    """
    cnv_matrix = np.asarray(cnv_matrix)
    ref_idx = np.asarray(reference_cell_idx)
    if ref_idx.dtype == bool:
        ref_vals = cnv_matrix[ref_idx, :]
    else:
        ref_vals = cnv_matrix[ref_idx.astype(np.int64), :]
    flat = ref_vals.astype(np.float64, copy=False).ravel()
    if flat.size < 2:
        raise ValueError(
            f"reference cells yielded {flat.size} values; need >= 2 for sd estimation"
        )

    mu = float(np.mean(flat))
    # R sd() uses N-1 denominator
    sigma = float(np.std(flat, ddof=1))
    if sigma <= 0.0:
        raise ValueError(f"reference sigma must be > 0, got {sigma}")

    # determine_mean_delta_via_Z: |qnorm(p, 0, sigma)| = sigma * |z_{p}|
    mean_delta = float(abs(norm.ppf(i3_p_val, loc=0.0, scale=sigma)))

    mus = np.array([mu - mean_delta, mu, mu + mean_delta], dtype=np.float64)
    sigmas = np.array([sigma, sigma, sigma], dtype=np.float64)
    return mus, sigmas


def predict_i3(  # noqa: N802
    cnv_matrix: np.ndarray,
    chr_pos: dict[str, int],
    *,
    transition_prob: float = 1e-6,
    state_mus: np.ndarray | None = None,
    state_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Return int8 array (n_cells, n_bins) of Viterbi-decoded 3-state CNV labels.

    States: 0 = DEL, 1 = neutral, 2 = AMP.

    Parameters mirror `predict_i6`. If `state_mus` / `state_sigmas` are None,
    they fall back to conservative proxies (symmetric around 0 in log2
    space). Callers should normally call `estimate_i3_state_params` first
    to fit params from the reference cells.
    """
    cnv_matrix = np.asarray(cnv_matrix)
    if cnv_matrix.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2-D, got shape {cnv_matrix.shape}")
    n_cells, n_bins = cnv_matrix.shape

    if state_mus is None:
        mus = np.array(
            [-I3_DEFAULT_MEAN_DELTA, 0.0, I3_DEFAULT_MEAN_DELTA], dtype=np.float64
        )
    else:
        mus = np.asarray(state_mus, dtype=np.float64).ravel()
    if mus.shape != (N_STATES_I3,):
        raise ValueError(f"state_mus must be shape ({N_STATES_I3},), got {mus.shape}")

    if state_sigmas is None:
        sigmas = np.full(N_STATES_I3, I3_DEFAULT_SIGMA, dtype=np.float64)
    else:
        sigmas = np.asarray(state_sigmas, dtype=np.float64).ravel()
    if sigmas.shape != (N_STATES_I3,):
        raise ValueError(f"state_sigmas must be shape ({N_STATES_I3},), got {sigmas.shape}")
    if np.any(sigmas <= 0.0):
        raise ValueError("state_sigmas must all be positive")

    log_trans = np.log(_build_transition_matrix(N_STATES_I3, transition_prob))
    neutral_idx = 1
    log_delta = np.log(_build_initial_distribution(N_STATES_I3, neutral_idx, transition_prob))

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
    "predict_i3",
    "estimate_i3_state_params",
    "N_STATES_I3",
    "I3_DEFAULT_SIGMA",
    "I3_DEFAULT_MEAN_DELTA",
]

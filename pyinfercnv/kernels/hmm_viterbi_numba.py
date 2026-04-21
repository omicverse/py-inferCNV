"""Log-space Viterbi decoder — numba-accelerated + numpy reference path.

Implements R `HiddenMarkov::dthmm` / `Viterbi.dthmm.adj` used in
`inferCNV_HMM.R:307-314, 345-408` and `inferCNV_i3HMM.R:208-215`. The R
`adj` variant does **not** use a Gaussian log-density for emissions;
it uses a custom "peakedness score" over upper-tail p-values. See
:func:`compute_log_emit` for the two supported variants:

  * ``emission='rstyle'`` (default) — bit-exact replica of R
    ``Viterbi.dthmm.adj`` (``inferCNV_HMM.R:1122-1162``) including the
    ``object$pm$sd <- median(object$pm$sd)`` override on line 1122.
    Selected by i3/i6 callers for R-parity.

  * ``emission='gauss_std'`` — standard Gaussian log-density
    ``log N(x | mu_k, sigma_k)``. Kept for cross-validation against
    hmmlearn's ``GaussianHMM.decode`` and as a general-purpose HMM
    tool. NOT used by the default i3/i6 pipeline.

Reference path (`viterbi_decode_numpy`) uses scipy.special.logsumexp in
`forward_backward_numpy` for the forward-backward pass. Viterbi DP is
pure numpy / numba argmax; both variants share the same DP so the only
bit-level divergence between emission modes is by construction inside
:func:`compute_log_emit`.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.stats import norm

_LOG_2PI = math.log(2.0 * math.pi)


def _gauss_log_emit(x: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Standard Gaussian log-density emission, shape (T, K).

    ``log N(x | mu_k, sigma_k) = -0.5 * [log(2*pi) + 2*log(sigma_k) + ((x - mu_k)/sigma_k)^2]``
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mus = np.asarray(mus, dtype=np.float64).ravel()
    sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
    T = x.shape[0]
    K = mus.shape[0]
    out = np.empty((T, K), dtype=np.float64)
    for k in range(K):
        inv_sig = 1.0 / sigmas[k]
        z = (x - mus[k]) * inv_sig
        out[:, k] = -0.5 * (_LOG_2PI + 2.0 * math.log(sigmas[k]) + z * z)
    return out


def _rstyle_log_emit(x: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """R ``Viterbi.dthmm.adj`` custom emission (inferCNV_HMM.R:1122-1162), shape (T, K).

    Replicates the R formula exactly::

        object$pm$sd <- median(object$pm$sd)          # line 1122, applies uniform sd
        emission[t,k] = pnorm(|(x_t - mu_k)/sd|, log.p=TRUE, lower.tail=FALSE)
                        -> this is log P(Z > |z|), always negative
        emission[t,k] = 1 / (-emission[t,k])          # invert magnitude, positive
        emission[t,k] = emission[t,k] / sum_k(emission[t,k])  # normalise to pmf over states
        log_emit[t,k] = log(emission[t,k])

    The ``median(sd)`` override is a no-op for i3 (shared sigma) but
    essential for i6 where hspike calibration yields per-state sigmas.
    Uses :func:`scipy.stats.norm.logsf` which has a stable asymptotic
    expansion for ``|z| >> 1`` (avoids underflow when naive
    ``log(0.5 * erfc(|z|/sqrt(2)))`` would lose precision).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mus = np.asarray(mus, dtype=np.float64).ravel()
    sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
    # R override: replace all sigmas with their median before emission
    sd_med = float(np.median(sigmas))
    z_abs = np.abs(x[:, None] - mus[None, :]) / sd_med   # (T, K)
    log_tail = norm.logsf(z_abs)                         # (T, K), < 0
    emit = 1.0 / (-log_tail)                             # (T, K), > 0
    emit = emit / emit.sum(axis=1, keepdims=True)        # normalise across K
    return np.log(emit)                                  # (T, K)


def compute_log_emit(
    obs: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    emission: str = "rstyle",
) -> np.ndarray:
    """Return (T, K) log-emission matrix for one of two models.

    Parameters
    ----------
    obs : (T,) float
    mus : (K,) float
    sigmas : (K,) float, strictly positive
    emission : {'rstyle', 'gauss_std'}
        ``'rstyle'`` (default) mirrors R ``Viterbi.dthmm.adj``; used by
        the default i3 / i6 decode paths for R-parity.
        ``'gauss_std'`` is the textbook Gaussian log-density; used only
        by cross-checks against external HMM libraries.
    """
    if emission == "rstyle":
        return _rstyle_log_emit(obs, mus, sigmas)
    if emission == "gauss_std":
        return _gauss_log_emit(obs, mus, sigmas)
    raise ValueError(
        f"emission must be 'rstyle' or 'gauss_std', got {emission!r}"
    )


def _viterbi_dp_numpy(
    log_emit: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
) -> np.ndarray:
    """Log-space Viterbi DP given precomputed emissions.

    Shapes::
        log_emit    (T, K)
        log_delta   (K,)
        log_trans   (K, K)   row i -> col j
    Returns int64 state path of length T. State indices 0..K-1.
    """
    T, K = log_emit.shape
    if T == 0:
        return np.empty(0, dtype=np.int64)

    V = np.empty((T, K), dtype=np.float64)
    bp = np.empty((T, K), dtype=np.int64)
    V[0, :] = log_delta + log_emit[0, :]
    bp[0, :] = 0
    for t in range(1, T):
        scores = V[t - 1, :, None] + log_trans   # (K, K)
        bp[t, :] = np.argmax(scores, axis=0)
        V[t, :] = scores[bp[t, :], np.arange(K)] + log_emit[t, :]

    path = np.empty(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(V[T - 1, :]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


@njit(cache=True, fastmath=False)
def _viterbi_dp_numba(
    log_emit: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
) -> np.ndarray:
    """Numba log-space Viterbi DP given precomputed emissions.

    Mirrors :func:`_viterbi_dp_numpy` argmax reduction order bit-for-bit
    (first-max-wins, identical to numpy's argmax with C-order inputs).
    """
    T = log_emit.shape[0]
    K = log_emit.shape[1]
    path = np.empty(T, dtype=np.int64)
    if T == 0:
        return path

    V_prev = np.empty(K, dtype=np.float64)
    V_curr = np.empty(K, dtype=np.float64)
    bp = np.empty((T, K), dtype=np.int64)

    for k in range(K):
        V_prev[k] = log_delta[k] + log_emit[0, k]
        bp[0, k] = 0

    for t in range(1, T):
        for j in range(K):
            best_i = 0
            best_s = V_prev[0] + log_trans[0, j]
            for i in range(1, K):
                s = V_prev[i] + log_trans[i, j]
                if s > best_s:
                    best_s = s
                    best_i = i
            V_curr[j] = best_s + log_emit[t, j]
            bp[t, j] = best_i
        for k in range(K):
            V_prev[k] = V_curr[k]

    best_last = 0
    best_v = V_prev[0]
    for k in range(1, K):
        if V_prev[k] > best_v:
            best_v = V_prev[k]
            best_last = k
    path[T - 1] = best_last
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def viterbi_decode_numpy(
    obs: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    *,
    emission: str = "rstyle",
) -> np.ndarray:
    """Reference numpy Viterbi with R-style emission by default.

    Shapes::
        obs         (T,)
        log_delta   (K,)        initial distribution, in log
        log_trans   (K, K)      transition, in log (row i -> col j)
        mus         (K,)        per-state means (gauss_std) or centres (rstyle)
        sigmas      (K,)        per-state sds — rstyle overrides to median
    emission : {'rstyle', 'gauss_std'}
        See :func:`compute_log_emit`. Default 'rstyle' for R-parity.
    Returns int64 state path of length T, state indices 0..K-1.
    """
    log_emit = compute_log_emit(obs, mus, sigmas, emission=emission)
    log_delta = np.asarray(log_delta, dtype=np.float64).ravel()
    log_trans = np.asarray(log_trans, dtype=np.float64)
    return _viterbi_dp_numpy(log_emit, log_delta, log_trans)


def viterbi_decode_numba(
    obs: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    *,
    emission: str = "rstyle",
) -> np.ndarray:
    """Numba log-space Viterbi with R-style emission by default.

    Same signature as :func:`viterbi_decode_numpy`; the only difference
    is the inner DP uses a compiled @njit kernel. Emissions are
    precomputed in numpy (``scipy.stats.norm.logsf`` is not njit-able);
    the DP loop is the only hot path and remains @njit.
    """
    log_emit = np.ascontiguousarray(
        compute_log_emit(obs, mus, sigmas, emission=emission),
        dtype=np.float64,
    )
    log_delta = np.ascontiguousarray(
        np.asarray(log_delta, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    log_trans = np.ascontiguousarray(
        np.asarray(log_trans, dtype=np.float64),
        dtype=np.float64,
    )
    return _viterbi_dp_numba(log_emit, log_delta, log_trans)


def forward_backward_numpy(
    obs: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    *,
    emission: str = "gauss_std",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-space forward-backward. Returns (log_alpha, log_beta, log_evidence).

    Default emission is ``'gauss_std'`` because forward-backward is only
    meaningful when emissions are true probability densities. R-style
    emissions are already row-normalised within-state and do not compose
    into a proper joint likelihood over time; callers who need R parity
    at the forward-backward level should revisit the math first.
    """
    from scipy.special import logsumexp

    obs = np.asarray(obs, dtype=np.float64).ravel()
    log_delta = np.asarray(log_delta, dtype=np.float64).ravel()
    log_trans = np.asarray(log_trans, dtype=np.float64)
    T = obs.shape[0]
    K = log_delta.shape[0]

    log_emit = compute_log_emit(obs, mus, sigmas, emission=emission)

    log_alpha = np.empty((T, K), dtype=np.float64)
    log_beta = np.empty((T, K), dtype=np.float64)
    if T == 0:
        return log_alpha, log_beta, 0.0

    log_alpha[0, :] = log_delta + log_emit[0, :]
    for t in range(1, T):
        log_alpha[t, :] = (
            logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0) + log_emit[t, :]
        )

    log_beta[T - 1, :] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t, :] = logsumexp(
            log_trans + (log_emit[t + 1, :] + log_beta[t + 1, :])[None, :], axis=1
        )

    log_evidence = float(logsumexp(log_alpha[T - 1, :]))
    return log_alpha, log_beta, log_evidence


__all__ = [
    "viterbi_decode_numpy",
    "viterbi_decode_numba",
    "forward_backward_numpy",
    "compute_log_emit",
]

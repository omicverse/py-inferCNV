"""Log-space Viterbi decoder — numba-accelerated + numpy reference path.

Mirrors `HiddenMarkov::dthmm` / `Viterbi.dthmm.adj` used in
`inferCNV_HMM.R:307-314` and `inferCNV_i3HMM.R:208-215`. Gaussian emissions
only (R uses `distn="norm"`).

Reference path (`viterbi_decode_numpy`) uses scipy.special.logsumexp-compatible
plain log arithmetic — kept for numerical-stability audit and bit-exact
correctness testing against the njit path. Both paths perform the same
reduction order (last-max-wins on ties, identical to numba's argmax and
numpy's argmax).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


_LOG_2PI = math.log(2.0 * math.pi)


def _gauss_log_emission(x: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Return (T, K) log-emission matrix for obs x and Gaussian state params.

    log N(x | mu_k, sigma_k) = -0.5 * [log(2*pi) + 2*log(sigma_k) + ((x - mu_k)/sigma_k)^2]
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


def viterbi_decode_numpy(
    obs: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Reference numpy Viterbi. Shapes:
        obs         (T,)
        log_delta   (K,)        initial distribution, in log
        log_trans   (K, K)      transition, in log (row i -> col j)
        mus         (K,)
        sigmas      (K,)

    Returns int64 state path of length T. State indices are 0..K-1.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    log_delta = np.asarray(log_delta, dtype=np.float64).ravel()
    log_trans = np.asarray(log_trans, dtype=np.float64)
    T = obs.shape[0]
    K = log_delta.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int64)

    log_emit = _gauss_log_emission(obs, mus, sigmas)  # (T, K)

    # DP matrices
    V = np.empty((T, K), dtype=np.float64)
    bp = np.empty((T, K), dtype=np.int64)

    V[0, :] = log_delta + log_emit[0, :]
    bp[0, :] = 0

    for t in range(1, T):
        # scores[i, j] = V[t-1, i] + log_trans[i, j]
        scores = V[t - 1, :, None] + log_trans  # (K, K)
        bp[t, :] = np.argmax(scores, axis=0)
        V[t, :] = scores[bp[t, :], np.arange(K)] + log_emit[t, :]

    path = np.empty(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(V[T - 1, :]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


@njit(cache=True, fastmath=False)
def viterbi_decode_numba(
    obs: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Numba log-space Viterbi; mirrors viterbi_decode_numpy reduction order.

    Inputs must be float64 1-D / 2-D arrays as in the numpy reference.
    Returns int64 path of length T.
    """
    T = obs.shape[0]
    K = log_delta.shape[0]
    path = np.empty(T, dtype=np.int64)
    if T == 0:
        return path

    V_prev = np.empty(K, dtype=np.float64)
    V_curr = np.empty(K, dtype=np.float64)
    bp = np.empty((T, K), dtype=np.int64)

    # t = 0
    for k in range(K):
        inv_sig = 1.0 / sigmas[k]
        z = (obs[0] - mus[k]) * inv_sig
        emit = -0.5 * (_LOG_2PI + 2.0 * math.log(sigmas[k]) + z * z)
        V_prev[k] = log_delta[k] + emit
        bp[0, k] = 0

    # t >= 1
    for t in range(1, T):
        for j in range(K):
            best_i = 0
            best_s = V_prev[0] + log_trans[0, j]
            for i in range(1, K):
                s = V_prev[i] + log_trans[i, j]
                if s > best_s:
                    best_s = s
                    best_i = i
            inv_sig = 1.0 / sigmas[j]
            z = (obs[t] - mus[j]) * inv_sig
            emit = -0.5 * (_LOG_2PI + 2.0 * math.log(sigmas[j]) + z * z)
            V_curr[j] = best_s + emit
            bp[t, j] = best_i
        # swap
        for k in range(K):
            V_prev[k] = V_curr[k]

    # backtrace
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


def forward_backward_numpy(
    obs: np.ndarray,
    log_delta: np.ndarray,
    log_trans: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-space forward-backward. Returns (log_alpha, log_beta, log_evidence).

    Uses scipy.special.logsumexp for numerical stability (matches the
    "mandatory logsumexp for log-space arithmetic" requirement).
    """
    from scipy.special import logsumexp

    obs = np.asarray(obs, dtype=np.float64).ravel()
    log_delta = np.asarray(log_delta, dtype=np.float64).ravel()
    log_trans = np.asarray(log_trans, dtype=np.float64)
    T = obs.shape[0]
    K = log_delta.shape[0]

    log_emit = _gauss_log_emission(obs, mus, sigmas)

    log_alpha = np.empty((T, K), dtype=np.float64)
    log_beta = np.empty((T, K), dtype=np.float64)
    if T == 0:
        return log_alpha, log_beta, 0.0

    log_alpha[0, :] = log_delta + log_emit[0, :]
    for t in range(1, T):
        # alpha[t, j] = logsumexp_i (alpha[t-1, i] + trans[i, j]) + emit[t, j]
        log_alpha[t, :] = (
            logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0) + log_emit[t, :]
        )

    log_beta[T - 1, :] = 0.0
    for t in range(T - 2, -1, -1):
        # beta[t, i] = logsumexp_j (trans[i, j] + emit[t+1, j] + beta[t+1, j])
        log_beta[t, :] = logsumexp(
            log_trans + (log_emit[t + 1, :] + log_beta[t + 1, :])[None, :], axis=1
        )

    log_evidence = float(logsumexp(log_alpha[T - 1, :]))
    return log_alpha, log_beta, log_evidence


__all__ = [
    "viterbi_decode_numpy",
    "viterbi_decode_numba",
    "forward_backward_numpy",
]

"""Unit tests for `pyinfercnv.hmm.i6`."""
from __future__ import annotations

import numpy as np
import pytest

from pyinfercnv.hmm import predict_i6
from pyinfercnv.hmm.i6 import I6_DEFAULT_MUS, I6_DEFAULT_SIGMA, N_STATES_I6
from pyinfercnv.kernels.hmm_viterbi_numba import (
    viterbi_decode_numba,
    viterbi_decode_numpy,
)


# --------------------------------------------------------------------------
# Determinism + shape + dtype
# --------------------------------------------------------------------------


def _small_chr_pos(n_bins: int) -> dict[str, int]:
    """Two-chromosome layout: chr1 first half, chr2 second half."""
    mid = n_bins // 2
    return {"chr1": 0, "chr2": mid}


def test_predict_i6_shape_and_dtype():
    n_cells, n_bins = 4, 30
    rng = np.random.default_rng(0)
    mat = rng.normal(0.0, 0.05, size=(n_cells, n_bins)).astype(np.float32)
    chr_pos = _small_chr_pos(n_bins)

    out = predict_i6(mat, chr_pos)

    assert out.shape == (n_cells, n_bins)
    assert out.dtype == np.int8
    assert out.min() >= 0 and out.max() < N_STATES_I6


def test_predict_i6_deterministic():
    rng = np.random.default_rng(123)
    mat = rng.normal(0.0, 0.1, size=(3, 20)).astype(np.float32)
    chr_pos = _small_chr_pos(20)

    out_a = predict_i6(mat.copy(), chr_pos)
    out_b = predict_i6(mat.copy(), chr_pos)
    np.testing.assert_array_equal(out_a, out_b)


# --------------------------------------------------------------------------
# Degenerate-signal cases
# --------------------------------------------------------------------------


def test_predict_i6_uniform_neutral_input_calls_neutral():
    """Centered ref-like input (all zeros in log2 space) -> neutral state (idx=2)."""
    mat = np.zeros((5, 40), dtype=np.float32)
    chr_pos = _small_chr_pos(40)
    out = predict_i6(mat, chr_pos)
    assert np.all(out == 2), f"expected all neutral (state=2), got counts {np.bincount(out.ravel(), minlength=6)}"


def test_predict_i6_strong_amplification_calls_amp_state():
    """Strong positive log2FC -> high-CN state (4 or 5)."""
    # Use well-separated mus/sigmas to force clean separation.
    mus = np.log2(np.array([0.01, 0.5, 1.0, 1.5, 2.0, 3.0]))
    sigmas = np.full(6, 0.1)
    # Strong AMP signal: input centered near log2(3.0) ≈ 1.585
    mat = np.full((3, 30), np.log2(3.0), dtype=np.float32)
    chr_pos = _small_chr_pos(30)
    out = predict_i6(mat, chr_pos, state_mus=mus, state_sigmas=sigmas)
    # State 5 corresponds to cnv=3.0 exactly.
    assert np.all(out == 5), f"expected all state=5 AMP, got {np.bincount(out.ravel(), minlength=6)}"


def test_predict_i6_strong_deletion_calls_del_state():
    """Strong negative log2FC -> low-CN state."""
    mus = np.log2(np.array([0.01, 0.5, 1.0, 1.5, 2.0, 3.0]))
    sigmas = np.full(6, 0.1)
    # Input near log2(0.5) ≈ -1 for heterozygous DEL.
    mat = np.full((3, 30), np.log2(0.5), dtype=np.float32)
    chr_pos = _small_chr_pos(30)
    out = predict_i6(mat, chr_pos, state_mus=mus, state_sigmas=sigmas)
    assert np.all(out == 1), f"expected all state=1 het-del, got {np.bincount(out.ravel(), minlength=6)}"


# --------------------------------------------------------------------------
# Per-chromosome segmentation
# --------------------------------------------------------------------------


def test_predict_i6_per_chromosome_independence():
    """State at chr2 boundary must NOT depend on chr1 content (independent sequences)."""
    mus = np.log2(np.array([0.01, 0.5, 1.0, 1.5, 2.0, 3.0]))
    sigmas = np.full(6, 0.1)
    n = 20
    chr_pos = {"chr1": 0, "chr2": n // 2}

    # chr1 strongly DEL, chr2 strongly AMP -> expect state 1 then state 5.
    mat = np.empty((1, n), dtype=np.float32)
    mat[0, : n // 2] = np.log2(0.5)
    mat[0, n // 2 :] = np.log2(3.0)
    out = predict_i6(mat, chr_pos, state_mus=mus, state_sigmas=sigmas)
    assert np.all(out[0, : n // 2] == 1)
    assert np.all(out[0, n // 2 :] == 5)


# --------------------------------------------------------------------------
# Numba vs numpy reference path
# --------------------------------------------------------------------------


def test_numba_vs_numpy_viterbi_bit_exact():
    """Numba log-space Viterbi must agree with numpy reference bit-exact."""
    rng = np.random.default_rng(42)
    T, K = 200, 6
    obs = rng.normal(0.0, 0.5, size=T)
    mus = I6_DEFAULT_MUS.copy()
    sigmas = np.full(K, I6_DEFAULT_SIGMA)
    t = 1e-6
    trans = np.full((K, K), t)
    np.fill_diagonal(trans, 1.0 - (K - 1) * t)
    delta = np.full(K, t)
    delta[2] = 1.0 - (K - 1) * t

    ref = viterbi_decode_numpy(obs, np.log(delta), np.log(trans), mus, sigmas)
    fast = viterbi_decode_numba(obs, np.log(delta), np.log(trans), mus, sigmas)

    np.testing.assert_array_equal(ref, fast)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_predict_i6_bad_transition_prob():
    mat = np.zeros((2, 10), dtype=np.float32)
    chr_pos = {"chr1": 0}
    with pytest.raises(ValueError, match="transition_prob"):
        predict_i6(mat, chr_pos, transition_prob=0.5)


def test_predict_i6_bad_state_mus_shape():
    mat = np.zeros((2, 10), dtype=np.float32)
    chr_pos = {"chr1": 0}
    with pytest.raises(ValueError, match="state_mus"):
        predict_i6(mat, chr_pos, state_mus=np.zeros(5))


# --------------------------------------------------------------------------
# Optional hmmlearn cross-check (skip if not installed)
# --------------------------------------------------------------------------


def test_hmmlearn_parity_optional():
    """Soft parity with hmmlearn GaussianHMM.decode (Viterbi)."""
    hmmlearn = pytest.importorskip("hmmlearn")
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(7)
    T = 300
    K = 6
    mus = I6_DEFAULT_MUS.copy()
    sigmas = np.full(K, 0.3)
    t = 1e-4
    obs = rng.normal(0.0, 0.3, size=T)  # noisy neutral-ish signal

    trans = np.full((K, K), t)
    np.fill_diagonal(trans, 1.0 - (K - 1) * t)
    delta = np.full(K, t)
    delta[2] = 1.0 - (K - 1) * t

    hm = GaussianHMM(n_components=K, covariance_type="diag", init_params="")
    hm.startprob_ = delta
    hm.transmat_ = trans
    hm.means_ = mus.reshape(-1, 1)
    hm.covars_ = (sigmas**2).reshape(-1, 1)
    _, states_hmml = hm.decode(obs.reshape(-1, 1), algorithm="viterbi")

    ours = viterbi_decode_numpy(
        obs, np.log(delta), np.log(trans), mus, sigmas,
        emission="gauss_std",  # hmmlearn uses standard Gaussian; R-style would diverge
    )
    agree = float(np.mean(ours == states_hmml))
    # Soft floor — hmmlearn's tie-break policy can differ at boundaries.
    assert agree >= 0.95, f"hmmlearn agreement only {agree:.3f}"

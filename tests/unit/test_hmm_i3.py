"""Unit tests for `pyinfercnv.hmm.i3`."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyinfercnv.hmm import estimate_i3_state_params, predict_i3
from pyinfercnv.hmm.i3 import N_STATES_I3
from pyinfercnv.kernels.hmm_viterbi_numba import (
    viterbi_decode_numba,
    viterbi_decode_numpy,
)


def _chr_pos(n_bins: int) -> dict[str, int]:
    return {"chr1": 0, "chr2": n_bins // 2}


# --------------------------------------------------------------------------
# predict_i3 basic contracts
# --------------------------------------------------------------------------


def test_predict_i3_shape_and_dtype():
    mat = np.zeros((3, 20), dtype=np.float32)
    out = predict_i3(mat, _chr_pos(20))
    assert out.shape == (3, 20)
    assert out.dtype == np.int8
    assert out.min() >= 0 and out.max() < N_STATES_I3


def test_predict_i3_deterministic():
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 0.1, (2, 40)).astype(np.float32)
    a = predict_i3(mat.copy(), _chr_pos(40))
    b = predict_i3(mat.copy(), _chr_pos(40))
    np.testing.assert_array_equal(a, b)


def test_predict_i3_uniform_neutral_input():
    mat = np.zeros((4, 30), dtype=np.float32)
    out = predict_i3(mat, _chr_pos(30))
    assert np.all(out == 1), f"expected all neutral (state=1), got counts {np.bincount(out.ravel(), minlength=3)}"


def test_predict_i3_strong_amplification():
    mat = np.full((3, 30), 0.8, dtype=np.float32)
    mus = np.array([-0.5, 0.0, 0.5])
    sigmas = np.full(3, 0.1)
    out = predict_i3(mat, _chr_pos(30), state_mus=mus, state_sigmas=sigmas)
    assert np.all(out == 2), f"expected all AMP (state=2), got {np.bincount(out.ravel(), minlength=3)}"


def test_predict_i3_strong_deletion():
    mat = np.full((3, 30), -0.8, dtype=np.float32)
    mus = np.array([-0.5, 0.0, 0.5])
    sigmas = np.full(3, 0.1)
    out = predict_i3(mat, _chr_pos(30), state_mus=mus, state_sigmas=sigmas)
    assert np.all(out == 0), f"expected all DEL (state=0), got {np.bincount(out.ravel(), minlength=3)}"


def test_predict_i3_per_chromosome_independence():
    n = 40
    mus = np.array([-0.5, 0.0, 0.5])
    sigmas = np.full(3, 0.1)
    mat = np.empty((1, n), dtype=np.float32)
    mat[0, : n // 2] = -0.8
    mat[0, n // 2 :] = 0.8
    out = predict_i3(mat, _chr_pos(n), state_mus=mus, state_sigmas=sigmas)
    assert np.all(out[0, : n // 2] == 0)
    assert np.all(out[0, n // 2 :] == 2)


def test_predict_i3_bad_transition_prob():
    mat = np.zeros((1, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="transition_prob"):
        predict_i3(mat, {"chr1": 0}, transition_prob=0.25)


# --------------------------------------------------------------------------
# estimate_i3_state_params
# --------------------------------------------------------------------------


def test_estimate_i3_state_params_shapes_and_symmetry():
    rng = np.random.default_rng(1)
    n_cells, n_bins = 50, 200
    mat = rng.normal(0.0, 0.3, (n_cells, n_bins)).astype(np.float32)
    ref_idx = np.arange(30)  # first 30 cells are reference

    mus, sigmas = estimate_i3_state_params(mat, ref_idx, i3_p_val=0.05)
    assert mus.shape == (3,)
    assert sigmas.shape == (3,)
    # Shared variance
    assert sigmas[0] == sigmas[1] == sigmas[2]
    # Symmetric around neutral mean
    np.testing.assert_allclose(mus[1] - mus[0], mus[2] - mus[1], rtol=1e-10)


def test_estimate_i3_state_params_matches_r_z_formula():
    """Verify the Z-formula exactly: mean_delta = |qnorm(p, 0, sigma)|."""
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, size=5000)
    mat = ref[:, None].repeat(10, axis=1).astype(np.float32)  # n_cells x n_bins
    ref_idx = np.arange(mat.shape[0])

    mus, sigmas = estimate_i3_state_params(mat, ref_idx, i3_p_val=0.05)

    expected_mu = float(np.mean(mat.astype(np.float64).ravel()))
    expected_sigma = float(np.std(mat.astype(np.float64).ravel(), ddof=1))
    expected_delta = float(abs(norm.ppf(0.05, loc=0.0, scale=expected_sigma)))

    np.testing.assert_allclose(sigmas[0], expected_sigma, rtol=1e-12)
    np.testing.assert_allclose(mus[1], expected_mu, rtol=1e-12)
    np.testing.assert_allclose(mus[0], expected_mu - expected_delta, rtol=1e-12)
    np.testing.assert_allclose(mus[2], expected_mu + expected_delta, rtol=1e-12)


def test_estimate_i3_state_params_boolean_mask():
    rng = np.random.default_rng(2)
    mat = rng.normal(0.0, 0.2, (10, 100)).astype(np.float32)
    mask = np.zeros(10, dtype=bool)
    mask[:5] = True
    mus_a, sigmas_a = estimate_i3_state_params(mat, mask, i3_p_val=0.05)
    mus_b, sigmas_b = estimate_i3_state_params(mat, np.arange(5), i3_p_val=0.05)
    np.testing.assert_allclose(mus_a, mus_b)
    np.testing.assert_allclose(sigmas_a, sigmas_b)


# --------------------------------------------------------------------------
# Numba / numpy bit-exact cross-check (K=3)
# --------------------------------------------------------------------------


def test_numba_vs_numpy_viterbi_i3_bit_exact():
    rng = np.random.default_rng(99)
    T, K = 250, 3
    obs = rng.normal(0.0, 0.4, size=T)
    mus = np.array([-0.5, 0.0, 0.5])
    sigmas = np.full(K, 0.3)
    t = 1e-6
    trans = np.full((K, K), t)
    np.fill_diagonal(trans, 1.0 - 5.0 * t)  # mirror R i3 exact `1-5t`
    delta = np.full(K, t)
    delta[1] = 1.0 - 5.0 * t

    ref = viterbi_decode_numpy(obs, np.log(delta), np.log(trans), mus, sigmas)
    fast = viterbi_decode_numba(obs, np.log(delta), np.log(trans), mus, sigmas)
    np.testing.assert_array_equal(ref, fast)


# --------------------------------------------------------------------------
# Optional hmmlearn cross-check
# --------------------------------------------------------------------------


def test_hmmlearn_parity_optional_i3():
    pytest.importorskip("hmmlearn")
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(13)
    T = 200
    K = 3
    mus = np.array([-0.5, 0.0, 0.5])
    sigmas = np.full(K, 0.2)
    t = 1e-4
    obs = rng.normal(0.0, 0.2, size=T)

    trans = np.full((K, K), t)
    np.fill_diagonal(trans, 1.0 - 5.0 * t)
    delta = np.full(K, t)
    delta[1] = 1.0 - 5.0 * t

    hm = GaussianHMM(n_components=K, covariance_type="diag", init_params="")
    hm.startprob_ = delta / delta.sum()  # hmmlearn insists on rowsum=1
    row_sum = trans.sum(axis=1, keepdims=True)
    hm.transmat_ = trans / row_sum
    hm.means_ = mus.reshape(-1, 1)
    hm.covars_ = (sigmas**2).reshape(-1, 1)
    _, states_hmml = hm.decode(obs.reshape(-1, 1), algorithm="viterbi")

    ours = viterbi_decode_numpy(
        obs, np.log(delta / delta.sum()), np.log(trans / row_sum), mus, sigmas,
        emission="gauss_std",  # hmmlearn uses standard Gaussian; R-style would diverge
    )
    agree = float(np.mean(ours == states_hmml))
    assert agree >= 0.95, f"i3 hmmlearn agreement {agree:.3f}"

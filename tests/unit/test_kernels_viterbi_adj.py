"""Bit-exact parity of py ``compute_log_emit(..., emission='rstyle')`` vs
R ``Viterbi.dthmm.adj`` (inferCNV_HMM.R:1101-1175).

Evidence produced by ``scripts/triage_i3/triage_r_viterbi.R`` — the R-side
synthetic dump lives under ``scripts/triage_i3/r_dump/``. If the TSVs are
absent the test skips (local R not required for CI).

Three assertions:
  1. py R-style log_emit matches R emissions TSV to 1e-12 per cell.
  2. py R-style Viterbi path matches R path exactly.
  3. py standard-Gaussian log_emit is DIFFERENT from R emissions (proves
     the two emission modes genuinely disagree, so the 'rstyle' claim
     is meaningful).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyinfercnv.kernels.hmm_viterbi_numba import (
    compute_log_emit,
    viterbi_decode_numba,
    viterbi_decode_numpy,
)

R_DUMP = Path(__file__).resolve().parents[2] / "scripts" / "triage_i3" / "r_dump"


def _r_dump_available() -> bool:
    required = ["input_x.tsv", "input_states.tsv", "input_trans.tsv",
                "input_delta.txt", "emissions.tsv", "path.txt"]
    return R_DUMP.exists() and all((R_DUMP / f).is_file() for f in required)


pytestmark = pytest.mark.skipif(
    not _r_dump_available(),
    reason=f"R triage dump not found under {R_DUMP}; run Rscript scripts/triage_i3/triage_r_viterbi.R",
)


def _load_r_dump():
    x = np.loadtxt(R_DUMP / "input_x.tsv", skiprows=1, usecols=(1,))
    states = np.loadtxt(R_DUMP / "input_states.tsv", skiprows=1, usecols=(1, 2))
    mus = states[:, 0]
    sigmas = states[:, 1]
    trans = np.loadtxt(R_DUMP / "input_trans.tsv")
    delta = np.array([float(s) for s in (R_DUMP / "input_delta.txt").read_text().split()])
    r_emissions = np.loadtxt(R_DUMP / "emissions.tsv")
    r_path = np.array([int(s) for s in (R_DUMP / "path.txt").read_text().split()])
    return x, mus, sigmas, trans, delta, r_emissions, r_path


def test_rstyle_log_emit_matches_r_bit_exact():
    """compute_log_emit(emission='rstyle') == R emissions (max diff < 1e-12)."""
    x, mus, sigmas, _, _, r_emissions, _ = _load_r_dump()
    py_log_emit = compute_log_emit(x, mus, sigmas, emission="rstyle")
    assert py_log_emit.shape == r_emissions.shape
    max_diff = float(np.max(np.abs(py_log_emit - r_emissions)))
    assert max_diff < 1e-12, f"R-style emission diff {max_diff:.3e} exceeds 1e-12"


def test_rstyle_viterbi_path_matches_r_exactly():
    """viterbi_decode_{numpy,numba} with emission='rstyle' == R path exactly."""
    x, mus, sigmas, trans, delta, _, r_path = _load_r_dump()
    log_delta = np.log(delta)
    log_trans = np.log(trans)

    path_numpy = viterbi_decode_numpy(x, log_delta, log_trans, mus, sigmas,
                                       emission="rstyle")
    path_numba = np.asarray(viterbi_decode_numba(x, log_delta, log_trans, mus, sigmas,
                                                   emission="rstyle"))

    np.testing.assert_array_equal(path_numpy, r_path)
    np.testing.assert_array_equal(path_numba, r_path)


def test_gauss_std_emission_differs_from_rstyle():
    """Sanity: 'gauss_std' and 'rstyle' produce different log_emit matrices.

    If they did not differ, the whole emission-mode split would be academic.
    We only check that the max element-wise diff exceeds a reasonable floor.
    """
    x, mus, sigmas, _, _, _, _ = _load_r_dump()
    emit_r = compute_log_emit(x, mus, sigmas, emission="rstyle")
    emit_g = compute_log_emit(x, mus, sigmas, emission="gauss_std")
    max_diff = float(np.max(np.abs(emit_r - emit_g)))
    # synthetic fixture has clear excursions; expect at least 1e-2 disagreement
    assert max_diff > 1e-2, f"emission modes suspiciously close ({max_diff:.3e})"


def test_median_sd_override_i6_like_sigmas():
    """R override ``pm$sd <- median(pm$sd)`` means the rstyle emission is
    insensitive to per-state sigma variation (critical for i6 where
    hspike yields unequal sigmas)."""
    rng = np.random.default_rng(3)
    T = 60
    K = 6
    x = rng.normal(0.0, 0.5, size=T)
    mus = np.linspace(-2.0, 2.0, K)
    # Two sigma scenarios with identical median but very different spread:
    #   sorted(sig_tight) = [0.2, 0.3, 0.5, 0.5, 0.7, 0.8] -> median 0.5
    #   sorted(sig_loose) = [0.1, 0.4, 0.5, 0.5, 0.6, 0.9] -> median 0.5
    sig_tight = np.array([0.2, 0.3, 0.5, 0.5, 0.7, 0.8])
    sig_loose = np.array([0.1, 0.4, 0.5, 0.5, 0.6, 0.9])
    assert float(np.median(sig_tight)) == float(np.median(sig_loose))
    emit_tight = compute_log_emit(x, mus, sig_tight, emission="rstyle")
    emit_loose = compute_log_emit(x, mus, sig_loose, emission="rstyle")
    np.testing.assert_allclose(emit_tight, emit_loose, rtol=0.0, atol=1e-15)

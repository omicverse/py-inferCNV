"""Unit tests for pyinfercnv.hmm.hspike — G2 Q6 shape/dtype invariants.

Synthetic fixtures, seeded RNG. No R execution required.
"""
from __future__ import annotations

import importlib
import inspect
import math

import numpy as np
import pytest

from pyinfercnv.hmm.hspike import (
    I6_CNV_LEVELS_CALIBRATED,
    calibrate_i6_emission,
)
from pyinfercnv.hmm import HspikeCalibration


# --------------------------------------------------------------------------- #
# Shared fixture                                                              #
# --------------------------------------------------------------------------- #

def _make_ref_counts(n_cells: int = 50, n_genes: int = 200, seed: int = 42) -> np.ndarray:
    """Synthetic reference count matrix with realistic positive values."""
    rng = np.random.default_rng(seed)
    # Negative binomial-like: mostly small positive integers
    counts = rng.negative_binomial(n=5, p=0.5, size=(n_cells, n_genes)).astype(np.float32)
    # Ensure no all-zero columns (would be filtered)
    counts[:, :10] += 2.0
    return counts


def _make_ref_groups(n_cells: int = 50) -> dict[str, np.ndarray]:
    half = n_cells // 2
    return {
        "groupA": np.arange(0, half, dtype=np.intp),
        "groupB": np.arange(half, n_cells, dtype=np.intp),
    }


# --------------------------------------------------------------------------- #
# Test: shapes and dtypes (G2 Q6)                                            #
# --------------------------------------------------------------------------- #

def test_hspike_calibration_shapes_and_dtypes():
    """Full run on 50 ref cells × 200 genes; check shape and dtype invariants."""
    ref_counts = _make_ref_counts(n_cells=50, n_genes=200)
    ref_groups = _make_ref_groups(50)

    calib = calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=10,
        num_genes_per_chr=10,
        trend_num_rounds=10,
        trend_max_num_cells=10,
        random_state=123,
        keep_matrix=False,
    )

    assert isinstance(calib, HspikeCalibration)

    assert calib.state_mus.shape == (6,), f"state_mus.shape={calib.state_mus.shape}"
    assert calib.state_mus.dtype == np.float64, f"state_mus.dtype={calib.state_mus.dtype}"

    assert calib.state_sigmas.shape == (6,), f"state_sigmas.shape={calib.state_sigmas.shape}"
    assert calib.state_sigmas.dtype == np.float64, f"state_sigmas.dtype={calib.state_sigmas.dtype}"

    assert calib.sd_log_slope.shape == (6,), f"sd_log_slope.shape={calib.sd_log_slope.shape}"
    assert calib.sd_log_slope.dtype == np.float64, f"sd_log_slope.dtype={calib.sd_log_slope.dtype}"

    assert calib.sd_log_intercept.shape == (6,), f"sd_log_intercept.shape={calib.sd_log_intercept.shape}"
    assert calib.sd_log_intercept.dtype == np.float64, f"sd_log_intercept.dtype={calib.sd_log_intercept.dtype}"

    assert calib.cnv_levels.tolist() == [0.01, 0.5, 1.0, 1.5, 2.0, 3.0], \
        f"cnv_levels={calib.cnv_levels.tolist()}"


# --------------------------------------------------------------------------- #
# Test: sigmas_for_num_cells shape                                            #
# --------------------------------------------------------------------------- #

def test_sigmas_for_num_cells_shape():
    """sigmas_for_num_cells returns shape (6,) float64, all > 0."""
    ref_counts = _make_ref_counts(50, 200)
    ref_groups = _make_ref_groups(50)

    calib = calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=10,
        num_genes_per_chr=10,
        trend_num_rounds=10,
        trend_max_num_cells=10,
        random_state=7,
    )
    sigs = calib.sigmas_for_num_cells(5)
    assert sigs.shape == (6,), f"shape={sigs.shape}"
    assert sigs.dtype == np.float64, f"dtype={sigs.dtype}"
    assert np.all(sigs > 0), f"not all > 0: {sigs}"


# --------------------------------------------------------------------------- #
# Test: monotone decreasing sd (central-limit)                               #
# --------------------------------------------------------------------------- #

def test_sigmas_for_num_cells_monotone_decreasing():
    """sd(100) <= sd(1) componentwise — central-limit scaling."""
    ref_counts = _make_ref_counts(50, 200)
    ref_groups = _make_ref_groups(50)

    calib = calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=20,
        num_genes_per_chr=15,
        trend_num_rounds=50,
        trend_max_num_cells=100,
        random_state=42,
    )
    sigs_1 = calib.sigmas_for_num_cells(1)
    sigs_100 = calib.sigmas_for_num_cells(100)
    assert np.all(sigs_100 <= sigs_1), (
        f"Expected sd(100) <= sd(1) but got:\n  sd(1)={sigs_1}\n  sd(100)={sigs_100}"
    )


# --------------------------------------------------------------------------- #
# Test: mu ordering for higher CN states                                      #
# --------------------------------------------------------------------------- #

def test_hspike_monotonic_order_in_mu():
    """state_mus[2] < state_mus[3] < state_mus[4] < state_mus[5] (higher CN -> higher log2-FC).

    state_mus[0] (cnv=0.01) is a floor; don't assert ordering for it.
    """
    ref_counts = _make_ref_counts(60, 200, seed=99)
    # Use a single ref group to keep it simple
    ref_groups = {"all": np.arange(60, dtype=np.intp)}

    calib = calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=30,
        num_genes_per_chr=20,
        trend_num_rounds=20,
        trend_max_num_cells=30,
        random_state=10,
    )
    mus = calib.state_mus
    # cnv=1.0 (idx2) < cnv=1.5 (idx3) < cnv=2.0 (idx4) < cnv=3.0 (idx5)
    assert mus[2] < mus[3], f"Expected mu[cnv=1.0]={mus[2]:.4f} < mu[cnv=1.5]={mus[3]:.4f}"
    assert mus[3] < mus[4], f"Expected mu[cnv=1.5]={mus[3]:.4f} < mu[cnv=2.0]={mus[4]:.4f}"
    assert mus[4] < mus[5], f"Expected mu[cnv=2.0]={mus[4]:.4f} < mu[cnv=3.0]={mus[5]:.4f}"


# --------------------------------------------------------------------------- #
# Test: unsupported sim_method                                                #
# --------------------------------------------------------------------------- #

def test_sim_method_unsupported_raises():
    """sim_method='simple' and 'splatter' raise NotImplementedError."""
    ref_counts = _make_ref_counts(20, 50)
    ref_groups = {"g": np.arange(20, dtype=np.intp)}

    with pytest.raises(NotImplementedError):
        calibrate_i6_emission(ref_counts, ref_groups, sim_method="simple")

    with pytest.raises(NotImplementedError):
        calibrate_i6_emission(ref_counts, ref_groups, sim_method="splatter")


# --------------------------------------------------------------------------- #
# Test: edge cases raise ValueError                                           #
# --------------------------------------------------------------------------- #

def test_edge_cases_raise():
    """Empty ref, group with 1 cell, num_cells_per_state=1 all raise ValueError."""
    good_counts = _make_ref_counts(10, 50)

    # Empty ref_counts_raw (0 cells)
    empty_counts = np.empty((0, 50), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        calibrate_i6_emission(empty_counts, None)

    # ref group with 1 cell
    with pytest.raises(ValueError, match="need >= 2"):
        calibrate_i6_emission(
            good_counts,
            {"single": np.array([0], dtype=np.intp)},
            num_cells_per_state=10,
        )

    # num_cells_per_state < 2
    with pytest.raises(ValueError, match="num_cells_per_state"):
        calibrate_i6_emission(
            good_counts,
            {"g": np.arange(10, dtype=np.intp)},
            num_cells_per_state=1,
        )


# --------------------------------------------------------------------------- #
# Test: keep_matrix default False                                             #
# --------------------------------------------------------------------------- #

def test_keep_matrix_default_false():
    """Default keep_matrix=False returns hspike_log2fc=None, gene_chr_labels=None."""
    ref_counts = _make_ref_counts(30, 100)
    ref_groups = {"g": np.arange(30, dtype=np.intp)}

    calib_no_matrix = calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=5,
        num_genes_per_chr=8,
        trend_num_rounds=5,
        trend_max_num_cells=10,
        random_state=0,
        keep_matrix=False,
    )
    assert calib_no_matrix.hspike_log2fc is None
    assert calib_no_matrix.gene_chr_labels is None

    calib_with_matrix = calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=5,
        num_genes_per_chr=8,
        trend_num_rounds=5,
        trend_max_num_cells=10,
        random_state=0,
        keep_matrix=True,
    )
    assert calib_with_matrix.hspike_log2fc is not None
    assert calib_with_matrix.gene_chr_labels is not None
    assert calib_with_matrix.hspike_log2fc.ndim == 2
    assert calib_with_matrix.hspike_log2fc.dtype == np.float32
    assert calib_with_matrix.gene_chr_labels.ndim == 1


# --------------------------------------------------------------------------- #
# Test: no pipeline circular import (G2 Q11)                                 #
# --------------------------------------------------------------------------- #

def test_no_pipeline_circular_import():
    """hspike.py must not import from pyinfercnv.pipeline or pyinfercnv.pipeline_phase2."""
    import pyinfercnv.hmm.hspike as hspike_mod
    source = inspect.getsource(hspike_mod)
    assert "from pyinfercnv.pipeline import" not in source, \
        "hspike.py imports from pyinfercnv.pipeline — circular import risk"
    assert "import pyinfercnv.pipeline" not in source, \
        "hspike.py imports pyinfercnv.pipeline — circular import risk"
    assert "from pyinfercnv.pipeline_phase2 import" not in source, \
        "hspike.py imports from pyinfercnv.pipeline_phase2 — circular import risk"
    assert "import pyinfercnv.pipeline_phase2" not in source, \
        "hspike.py imports pyinfercnv.pipeline_phase2 — circular import risk"


# --------------------------------------------------------------------------- #
# Test: profile hooks populated (G1-P11)                                     #
# --------------------------------------------------------------------------- #

def test_profile_hooks_populated():
    """Pass profile={}, verify 4 expected keys present with wallclock_s numeric."""
    ref_counts = _make_ref_counts(30, 100)
    ref_groups = {"g": np.arange(30, dtype=np.intp)}
    prof: dict = {}

    calibrate_i6_emission(
        ref_counts,
        ref_groups,
        num_cells_per_state=5,
        num_genes_per_chr=8,
        trend_num_rounds=5,
        trend_max_num_cells=10,
        random_state=1,
        profile=prof,
    )

    expected_keys = [
        "hspike_sim",
        "hspike_phase1_replay",
        "hspike_get_dists",
        "hspike_trend_lm",
    ]
    for key in expected_keys:
        assert key in prof, f"profile missing key {key!r}; got keys: {list(prof.keys())}"
        assert "wallclock_s" in prof[key], f"profile[{key!r}] missing 'wallclock_s'"
        assert isinstance(prof[key]["wallclock_s"], float), \
            f"profile[{key!r}]['wallclock_s'] is not float: {type(prof[key]['wallclock_s'])}"
        assert prof[key]["wallclock_s"] >= 0.0

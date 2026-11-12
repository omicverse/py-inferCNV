"""R-generated synthetic oracles for sampling, weighted GCV and input order."""
from pathlib import Path

import numpy as np
import pytest

from pyinfercnv import InferCNVConfig, infercnv
from pyinfercnv.hmm.hspike import calibrate_i6_emission
from pyinfercnv.hmm.r_compat import RCompatibleRNG, fit_r_smoothing_spline

FIXTURES = Path(__file__).with_name("fixtures")


def test_r_rng_mixed_stream_and_rejection_consumption():
    oracle = np.genfromtxt(FIXTURES / "hspike_rng.tsv", names=True)
    rng = RCompatibleRNG(20261022)
    np.testing.assert_allclose(rng.random(64), oracle["u"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(rng.standard_normal((8, 8)).ravel(), oracle["z"], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(rng.choice(4056, 64), oracle["idx"])
    np.testing.assert_array_equal(rng.choice(1, 64), oracle["singleton"])
    # This also detects skipped n=1 draws and over-consumption by rejection.
    np.testing.assert_allclose(rng.random(64), oracle["tail"], rtol=0, atol=1e-15)


def test_r_weighted_gcv_and_linear_extrapolation():
    data = np.genfromtxt(FIXTURES / "hspike_spline_input.tsv", names=True)
    expected = np.genfromtxt(FIXTURES / "hspike_spline_prediction.tsv", names=True)
    params = np.genfromtxt(FIXTURES / "hspike_spline_fit.tsv", names=True)
    fit = fit_r_smoothing_spline(data["x"], data["y"])
    np.testing.assert_allclose(fit(expected["x"]), expected["y"], rtol=0, atol=5e-4)
    assert abs(fit.spar - params["spar"]) < 1e-3
    assert abs(fit.df - params["df"]) < 0.05


@pytest.mark.parametrize("x", [np.ones(10), np.tile([1, 2, 3], 4)])
def test_unestimable_spline_raises(x):
    with pytest.raises(ValueError, match="IQR|four distinct"):
        fit_r_smoothing_spline(x, np.arange(len(x)))


def test_pipeline_calibration_order_precision_and_real_matrix(small_synthetic_adata):
    adata = small_synthetic_adata[:, ::-1].copy()
    native = infercnv(adata, config=InferCNVConfig(cutoff=0), inplace=False)
    legacy = infercnv(adata, config=InferCNVConfig(cutoff=0, hspike_calibration_backend="numpy_legacy"), inplace=False)
    raw = adata.layers["counts"].toarray().astype(np.float64)
    expected = raw * (np.median(raw.sum(axis=1)) / raw.sum(axis=1))[:, None]
    order = sorted(range(adata.n_vars), key=lambda i: (str(adata.var.chromosome.iloc[i]), adata.var.start.iloc[i]))
    assert native.cpm_matrix_f32.dtype == np.float64
    np.testing.assert_allclose(native.cpm_matrix_f32, expected[:, order], rtol=1e-15)
    np.testing.assert_allclose(legacy.cpm_matrix_f32, expected.astype(np.float32), rtol=1e-6)
    np.testing.assert_array_equal(native.cnv_matrix, legacy.cnv_matrix)


def test_reference_mapping_insertion_order_does_not_change_native_calibration():
    counts = np.random.default_rng(11).negative_binomial(5, 0.4, size=(40, 120)).astype(float)
    groups = {"B": np.arange(20), "A": np.arange(20, 40)}
    options = dict(num_cells_per_state=5, num_genes_per_chr=10, trend_num_rounds=5,
                   trend_max_num_cells=6, random_state=20261022)
    first = calibrate_i6_emission(counts, groups, **options)
    second = calibrate_i6_emission(counts, dict(reversed(list(groups.items()))), **options)
    np.testing.assert_array_equal(first.state_mus, second.state_mus)
    np.testing.assert_array_equal(first.sd_log_intercept, second.sd_log_intercept)

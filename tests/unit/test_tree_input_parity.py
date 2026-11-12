"""Regression tests for the count-to-tree handoff and R runmean semantics."""
import warnings

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from pyinfercnv import InferCNVConfig, infercnv
from pyinfercnv.preprocess.subtract_ref import subtract_reference
from pyinfercnv.subcluster import random_trees


def _counts_adata():
    counts = np.random.default_rng(31).integers(1, 30, size=(12, 24)).astype(float)
    adata = AnnData(sparse.csr_matrix(counts))
    adata.layers["counts"] = adata.X.copy()
    adata.obs_names = [f"cell_{i}" for i in range(12)]
    adata.var_names = [f"gene_{i}" for i in range(24)]
    adata.var["chromosome"] = ["chr2", "chr10", "chr1"] * 8
    adata.var["start"] = np.arange(24)[::-1] * 100
    adata.obs["celltype"] = ["normal_a"] * 3 + ["normal_b"] * 3 + ["tumor"] * 6
    return adata


def _expected_tree_input(adata):
    counts = adata.layers["counts"].toarray()
    normalized = counts * (np.median(counts.sum(axis=1)) / counts.sum(axis=1))[:, None]
    logged = np.log2(normalized + 1)
    group_means = np.log2(np.stack([normalized[:3].mean(axis=0),
                                   normalized[3:6].mean(axis=0)]) + 1)
    expected = logged - np.clip(logged, group_means.min(axis=0), group_means.max(axis=0))
    order = sorted(range(adata.n_vars),
                   key=lambda i: (adata.var.chromosome.iloc[i], adata.var.start.iloc[i]))
    return expected[:, order]


def test_excluded_chromosomes_do_not_change_normalization():
    base = _counts_adata()
    augmented = AnnData(sparse.hstack([base.X, np.full((12, 1), 1e6)], format="csr"))
    augmented.obs = base.obs.copy()
    augmented.var_names = list(base.var_names) + ["excluded_gene"]
    augmented.var["chromosome"] = list(base.var.chromosome) + ["chrX"]
    augmented.var["start"] = list(base.var.start) + [0]
    augmented.layers["counts"] = augmented.X.copy()
    cfg = InferCNVConfig(cutoff=0, min_cells_per_gene=0, window_length=3)
    options = dict(config=cfg, reference_key="celltype",
                   reference_cat=["normal_a", "normal_b"], inplace=False)
    expected = infercnv(base, **options)
    actual = infercnv(augmented, **options)
    np.testing.assert_array_equal(actual.cnv_matrix, expected.cnv_matrix)
    np.testing.assert_array_equal(actual.cpm_matrix_f32, expected.cpm_matrix_f32)
    assert actual.chr_pos == expected.chr_pos


@pytest.mark.parametrize("order", ["r_lexicographic", "natural"])
def test_step04_tree_handoff_uses_linear_reference_means(monkeypatch, order):
    adata = _counts_adata()
    captured = []

    def capture_tree(matrix, **kwargs):
        captured.append((matrix.copy(), kwargs))
        return np.zeros(len(matrix), dtype=np.int32)

    monkeypatch.setattr(random_trees, "random_tree_subcluster", capture_tree)
    cfg = InferCNVConfig(HMM=True, HMM_type="i3", cutoff=0, min_cells_per_gene=0,
                         window_length=3, tumor_subcluster_partition_method="random_trees",
                         tree_gene_order=order, tumor_subcluster_n_rand_iters=7)
    infercnv(adata, config=cfg, reference_key="celltype",
             reference_cat=["normal_a", "normal_b"], inplace=False)
    expected = _expected_tree_input(adata)
    if order == "natural":
        # The fixture has three chromosomes, each with eight genes.
        expected = expected[:, np.r_[0:8, 16:24, 8:16]]
    for (matrix, kwargs), rows in zip(captured, [slice(0, 3), slice(3, 6), slice(6, 12)], strict=True):
        np.testing.assert_allclose(matrix, expected[rows], rtol=0, atol=1e-14)
        assert kwargs["subsample_for_tree"] is None
        assert kwargs["smooth_method"] == "runmean"
        assert kwargs["n_rand_iters"] == 7


def test_inv_log_single_reference_uses_linear_mean():
    logged = np.log2(np.array([[1., 3.], [7., 1.], [15., 7.]]) + 1)
    actual = subtract_reference(logged, ref_groups={"normal": [0, 1]}, inv_log=True)
    expected = logged - np.log2(np.array([4., 2.]) + 1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("window", [3, 5, 101])
def test_tree_runmean_uses_cropped_endpoint_windows(window):
    matrix = np.array([[3., 1., 9., 2., 8., 4., 7.], [0., 2., 1., 5., 4., 6., 9.]])
    width = min(window, matrix.shape[1])
    half = width // 2
    expected = np.stack([matrix[:, max(0, i - half):i + half + 1].mean(axis=1)
                         for i in range(matrix.shape[1])], axis=1).astype(np.float32)
    expected = expected.astype(float) - np.median(expected, axis=1, keepdims=True)
    np.testing.assert_array_equal(random_trees._smooth_and_center(matrix, window), expected)


def test_random_tree_default_fits_all_cells(monkeypatch):
    seen = []

    def capture_null(matrix, **kwargs):
        seen.append(len(matrix))
        return np.empty((0, 4)), np.empty(0), 1.0

    monkeypatch.setattr(random_trees, "_parameterize_null", capture_null)
    with warnings.catch_warnings(record=True) as caught:
        labels = random_trees.random_tree_subcluster(np.ones((501, 5)))
    assert seen == [501]
    assert labels.shape == (501,)
    assert not any("subsample" in str(w.message) for w in caught)


def test_direct_phase2_requires_step04_tree_input():
    from pyinfercnv.pipeline_phase2 import _run_subclustering

    cfg = InferCNVConfig(tumor_subcluster_partition_method="random_trees")
    with pytest.raises(ValueError, match="requires tree_input"):
        _run_subclustering(np.ones((6, 10)), np.zeros(6, dtype=bool),
                           config=cfg, random_state=0, profile=None)

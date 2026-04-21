"""Tests for pyinfercnv.config.InferCNVConfig."""
from __future__ import annotations

import pytest
from pyinfercnv.config import InferCNVConfig


def test_default_construction():
    cfg = InferCNVConfig()
    assert cfg.cutoff == 1.0
    assert cfg.min_cells_per_gene == 3
    assert cfg.window_length == 101
    assert cfg.smooth_method == "pyramidinal"
    assert cfg.ref_subtract_use_mean_bounds is True
    assert cfg.max_centered_threshold == 3.0
    assert cfg.scale_data is False
    assert cfg.counts_layer == "counts"


def test_r_kwargs_preserved():
    cfg = InferCNVConfig(HMM_transition_prob=1e-5, BayesMaxPNormal=0.3)
    assert cfg.HMM_transition_prob == pytest.approx(1e-5)
    assert cfg.BayesMaxPNormal == pytest.approx(0.3)


def test_validate_window_length_odd():
    with pytest.raises(ValueError, match="window_length"):
        InferCNVConfig(window_length=100).validate()


def test_validate_cutoff_positive():
    with pytest.raises(ValueError, match="cutoff"):
        InferCNVConfig(cutoff=-1).validate()


def test_validate_passes_defaults():
    InferCNVConfig().validate()


def test_smooth_method_enum():
    with pytest.raises(ValueError, match="smooth_method"):
        InferCNVConfig(smooth_method="invalid").validate()


def test_bayes_max_p_normal_range():
    with pytest.raises(ValueError, match="BayesMaxPNormal"):
        InferCNVConfig(BayesMaxPNormal=1.5).validate()


def test_tumor_subcluster_partition_method_default():
    """G2 Q2: default is 'leiden'."""
    cfg = InferCNVConfig()
    assert cfg.tumor_subcluster_partition_method == "leiden"


def test_tumor_subcluster_partition_method_invalid_raises():
    """G2 Q2: invalid method raises ValueError."""
    with pytest.raises(ValueError, match="tumor_subcluster_partition_method"):
        InferCNVConfig(tumor_subcluster_partition_method="kmeans").validate()

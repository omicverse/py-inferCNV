"""Tests for pyinfercnv.viz.heatmap."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pyinfercnv.result import InferCNVResult  # noqa: E402
from pyinfercnv.viz.heatmap import plot_cnv_delta, plot_cnv_heatmap, plot_cnv_heatmap_compare  # noqa: E402


def _mk_result(seed: int = 0):
    chr_pos = {"chr1": 0, "chr2": 50}
    cnv = np.random.default_rng(seed).normal(0, 0.2, (20, 100)).astype(np.float32)
    fc = np.exp2(cnv)
    meta = pd.DataFrame(
        {"is_reference": [True] * 10 + [False] * 10},
        index=[f"c{i}" for i in range(20)],
    )
    return InferCNVResult(chr_pos=chr_pos, cnv_matrix=cnv, cnv_matrix_fc=fc, cell_meta=meta)


def test_plot_cnv_heatmap_returns_axes():
    ax = plot_cnv_heatmap(_mk_result())
    assert ax is not None
    plt.close("all")


def test_plot_does_not_import_omicverse():
    import re

    import pyinfercnv.viz.heatmap as m
    src = open(m.__file__).read()
    # Check for actual import statements, not docstring mentions.
    assert not re.search(r"^\s*(import|from)\s+omicverse", src, re.MULTILINE)


def test_plot_compare_returns_figure():
    fig = plot_cnv_heatmap_compare(_mk_result(0), _mk_result(1))
    assert fig is not None
    plt.close("all")


def test_plot_delta_returns_axes():
    ax = plot_cnv_delta(_mk_result(0), _mk_result(1))
    assert ax is not None
    plt.close("all")

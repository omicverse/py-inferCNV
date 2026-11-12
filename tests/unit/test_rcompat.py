"""R pieces reproduced by pyinfercnv._rcompat, against stored R output (tests/data/rcompat)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyinfercnv._rcompat.rng import RIgraphRNG, RRandom
from pyinfercnv._rcompat.spline import SmoothSpline

REF = Path(__file__).resolve().parent.parent / "data" / "rcompat"


def test_rng_stream_matches_r():
    """set.seed + rnorm (inversion, sd = 0 draws nothing) + runif + sample(replace = TRUE)."""
    r = RRandom(11)
    got = np.concatenate([
        r.norm_rand(2000),
        r.rnorm(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0, 2.0])),
        r.unif_rand(10),
        r.sample(100000, 50, replace=True) + 1.0,
    ])
    assert np.array_equal(got, np.fromfile(REF / "rng.bin"))


def test_smooth_spline_gcv_matches_r():
    """smooth.spline(x, y) with spar chosen by GCV, and predict() inside and outside the data."""
    xy = np.fromfile(REF / "spline_xy.bin")
    x, y = xy[: xy.size // 2], xy[xy.size // 2:]
    ref = np.fromfile(REF / "spline_fit.bin")
    f = SmoothSpline(x, y, spar=None)
    g = np.linspace(-0.5, x.max() + 0.5, 301)
    assert f.spar == ref[0]
    assert f.lambda_ == ref[1]
    assert np.array_equal(f.predict(g), ref[2:])


def test_leiden_on_r_generator_matches_r():
    """igraph Leiden drawing from R's generator returns R igraph's partition."""
    ig = pytest.importorskip("igraph")
    e = np.loadtxt(REF / "leiden_edges.tsv")
    g = ig.Graph(n=120, edges=(e[:, :2].astype(int) - 1).tolist(), directed=False)
    ig.set_random_number_generator(RIgraphRNG(RRandom(7)))
    try:
        p = g.community_leiden(objective_function="CPM", weights=e[:, 2].tolist(), resolution=0.08,
                               beta=0.01, n_iterations=2)
    finally:
        ig.set_random_number_generator(None)
    assert np.array_equal(np.asarray(p.membership) + 1, np.loadtxt(REF / "leiden_membership.txt", dtype=int))

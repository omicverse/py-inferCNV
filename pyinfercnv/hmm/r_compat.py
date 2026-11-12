"""Numerical primitives for R-compatible hspike calibration, without an R runtime.

Contracts: R's documented MT/Inversion/Rejection RNG and smooth.spline's
weighted cubic B-spline objective, duplicate pooling, knots and GCV. These
are independent NumPy/SciPy implementations, verified against R fixtures.
References: R manuals Random and smooth.spline; R RNG.c, snorm.c, sbart.c.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import ndtri


class RCompatibleRNG:
    """Local MT stream with R seed initialization and sampling conventions."""

    def __init__(self, seed: int | None):
        if seed is None:
            seed = np.random.SeedSequence().generate_state(1)[0]
        value = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            value = (69069 * value + 1) & 0xFFFFFFFF
        state = []
        for _ in range(625):
            value = (69069 * value + 1) & 0xFFFFFFFF
            state.append(value)
        self.bit_generator = np.random.MT19937(0)
        self.bit_generator.state = {"bit_generator": "MT19937", "state": {
            "key": np.asarray(state[1:], dtype=np.uint32), "pos": 624}}

    def random(self, size):
        values = self.bit_generator.random_raw(size).astype(np.float64) / 4294967296.0
        return np.maximum(values, 0.5 / 4294967295.0)

    def standard_normal(self, size):
        shape = (size,) if np.isscalar(size) else tuple(size)
        u = self.random((int(np.prod(shape)), 2))
        probabilities = (np.floor(134217728.0 * u[:, 0]) + u[:, 1]) / 134217728.0
        return ndtri(probabilities).reshape(shape)

    def choice(self, population, size, replace=True):
        if not replace:
            raise NotImplementedError("hspike only uses sampling with replacement")
        n = int(population) if np.isscalar(population) else len(population)
        if n < 1:
            raise ValueError("Sampling population must be positive")
        shape = (size,) if np.isscalar(size) else tuple(size)
        count = int(np.prod(shape))
        bits = (n - 1).bit_length()
        chunks = bits // 16 + 1
        indices = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            words = np.floor(self.random((count - filled, chunks)) * 65536).astype(np.uint64)
            candidate = np.zeros(len(words), dtype=np.uint64)
            for j in range(chunks):
                candidate = candidate * np.uint64(65536) + words[:, j]
            candidate &= np.uint64((1 << bits) - 1)
            accepted = candidate[candidate < n]
            indices[filled:filled + len(accepted)] = accepted
            filled += len(accepted)
        indices = indices.reshape(shape)
        return indices if np.isscalar(population) else np.asarray(population)[indices]


@dataclass(frozen=True)
class RSmoothingSpline:
    basis: BSpline
    minimum: float
    span: float
    spar: float
    df: float

    def __call__(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.minimum) / self.span
        prediction = self.basis(np.clip(z, 0.0, 1.0))
        prediction = np.where(z < 0, self.basis(0.0) + z * self.basis(0.0, nu=1), prediction)
        return np.where(z > 1, self.basis(1.0) + (z - 1.0) * self.basis(1.0, nu=1), prediction)


def fit_r_smoothing_spline(x, y):
    """Fit the default R smooth.spline objective with pooled weights and GCV.

    Constant input has no estimable curve and raises, as R does. No silent
    polynomial/interpolation fallback changes the calibration model.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spline inputs must be finite, equal-length vectors")
    tolerance = 1e-6 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))
    if tolerance <= 0:
        raise ValueError("R smoothing spline requires a positive x IQR")
    order = np.argsort(x, kind="stable")
    bins = np.rint((x[order] - x.mean()) / tolerance)
    first = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    ux = x[order][first]
    weights = np.diff(np.r_[first, len(x)]).astype(float)
    uy = np.add.reduceat(y[order], first) / weights
    within_ss = float(np.sum(np.add.reduceat(y[order] ** 2, first) - weights * uy ** 2))
    n = len(ux)
    if n < 4:
        raise ValueError("R smoothing spline requires four distinct x values")
    if n < 50:
        nknots = n
    elif n < 200:
        nknots = int(50 * 2 ** ((n - 50) / 150))
    elif n < 800:
        nknots = int(100 * (1.4 ** ((n - 200) / 600)))
    elif n < 3200:
        nknots = int(140 * ((200 / 140) ** ((n - 800) / 2400)))
    else:
        nknots = int(200 + (n - 3200) ** 0.2)
    z = (ux - ux[0]) / (ux[-1] - ux[0])
    indices = np.linspace(0, n - 1, nknots).astype(int)
    knots = np.r_[np.zeros(3), z[indices], np.ones(3)]
    ncoef = len(knots) - 4
    design = BSpline.design_matrix(z, knots, 3)
    gram = (design.T @ design.multiply(weights[:, None])).toarray()
    rhs = np.asarray(design.T @ (weights * uy))

    # R's sgram.f uses 0.3330, not 1/3, for the squared linear term.
    # Preserve that historical approximation: exact quadrature changes the
    # fitted endpoint slopes enough to affect extrapolation and rounding.
    breaks = np.unique(knots)
    widths = np.diff(breaks)
    basis = BSpline(knots, np.eye(ncoef), 3)
    left = basis(breaks[:-1], nu=2)
    change = basis(breaks[1:], nu=2) - left
    weighted = widths[:, None]
    cross = left.T @ (change * weighted)
    penalty = left.T @ (left * weighted) + (cross + cross.T) / 2 + 0.3330 * change.T @ (change * weighted)
    ratio = np.diag(gram)[2:-3].sum() / np.diag(penalty)[2:-3].sum()

    last_fit = None

    def solve(spar):
        nonlocal last_fit
        lam = ratio * 256 ** (3 * spar - 1)
        try:
            factor = cho_factor(gram + lam * penalty, check_finite=False)
            coef = cho_solve(factor, rhs, check_finite=False)
            df = float(np.trace(cho_solve(factor, gram, check_finite=False)))
        except np.linalg.LinAlgError:
            return np.inf
        last_fit = coef, df
        residual = uy - design @ coef
        rss = within_ss + np.dot(weights, residual ** 2)
        return rss / weights.sum() / (1 - df / weights.sum()) ** 2

    optimum = minimize_scalar(solve, bounds=(-1.5, 1.5), method="bounded",
                              options={"xatol": 1e-4, "maxiter": 500})
    if not optimum.success or not np.isfinite(optimum.fun) or last_fit is None:
        raise ValueError("R-compatible spline GCV did not converge")
    # R sbart reports the best spar but leaves coefficients/df from its
    # last evaluated candidate. Re-solving at the optimum changes the curve.
    coef, df = last_fit
    return RSmoothingSpline(BSpline(knots, coef, 3), float(ux[0]), float(ux[-1] - ux[0]), float(optimum.x), df)

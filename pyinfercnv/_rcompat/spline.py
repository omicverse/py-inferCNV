"""R ``smooth.spline(x, y, spar = ...)`` and its ``predict`` method.

scCDC fits the expected entropy-vs-expression curve with ``smooth.spline(spar = 1)``. This module
follows R 4.5's ``stats`` code step by step so the fitted values match R to rounding:

* the R front end (``smooth.spline``): ties in ``x`` are pooled after rounding ``(x - mean(x)) / tol``
  with ``tol = 1e-6 * IQR(x)``; ``x`` is mapped to [0, 1]; the knots are the pooled points at
  ``seq.int(1, nx, length.out = .nknots.smspl(nx))``;
* ``sbart.c``: the ``X'WX`` and penalty bands (``stxwx.f``, ``sgram.f``, including its
  ``.3330`` constant in place of 1/3), ``lambda = ratio * 16^(6 spar - 2)`` with ``ratio`` the ratio of
  the band traces, and the banded Cholesky solve (LINPACK ``dpbfa``/``dpbsl``) in ``sslvrg.f``;
* ``bvalue.f`` for evaluation, with R's linear extrapolation outside the data range.

B-spline values (de Boor's ``bsplvb``/``bsplvd``) and ``bvalue`` are evaluated for all points at
once; only the Cholesky factorisation of the ``nk x nk`` band (``nk`` ~ 200) runs as a scalar loop,
as in LINPACK.
"""
from __future__ import annotations

import math

import numpy as np

from .rbase import iqr, r_mean

__all__ = ["SmoothSpline", "smooth_spline"]

_K = 4
_THIRD = 0.333                     # sgram.f: ``.3330d0``


def _nknots(n: int) -> int:
    """``stats:::.nknots.smspl``."""
    if n < 50:
        return n
    a1, a2, a3, a4 = math.log2(50), math.log2(100), math.log2(140), math.log2(200)
    if n < 200:
        v = 2 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        v = 2 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        v = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        v = 200 + (n - 3200) ** 0.2
    return int(math.trunc(v))


def _seq_length_out(start: float, stop: float, n: int) -> np.ndarray:
    """``seq.int(start, stop, length.out = n)`` (R 4.5 ``do_seq``: ``start + i * by``)."""
    out = np.empty(n)
    if n > 0:
        out[0] = start
    if n > 1:
        out[-1] = stop
    if n > 2:
        by = (stop - start) / (n - 1)
        out[1:-1] = start + np.arange(1, n - 1) * by
    return out


def _bsplvb_init(x):
    return {"j": 1, "b": {1: np.ones_like(x)}, "dl": {}, "dr": {}}


def _bsplvb(t, x, left, jhigh, st):
    """de Boor's ``bsplvb``: raise the order of the values in ``st`` to ``jhigh``.

    ``t`` is 0-based; ``left`` holds the 1-based Fortran interval index for every point."""
    b, dl, dr = st["b"], st["dl"], st["dr"]
    j = st["j"]
    while j < jhigh:
        jp1 = j + 1
        dr[j] = t[left + j - 1] - x
        dl[j] = x - t[left - j]
        saved = 0.0
        for i in range(1, j + 1):
            term = b[i] / (dr[i] + dl[jp1 - i])
            b[i] = saved + dr[i] * term
            saved = dl[jp1 - i] * term
        b[jp1] = saved
        j = jp1
    st["j"] = j
    return st


def _bsplvd(t, x, left, nderiv):
    """de Boor's ``bsplvd`` for cubic splines: values and derivatives up to ``nderiv - 1``.

    Returns ``d`` with ``d[(i, m)]`` the (m-1)-th derivative of the i-th non-zero B-spline."""
    k, kp1 = _K, _K + 1
    mhigh = max(min(nderiv, k), 1)
    st = _bsplvb(t, x, left, kp1 - mhigh, _bsplvb_init(x))
    col1 = st["b"]
    d = {}
    if mhigh == 1:
        return {(i, 1): col1[i] for i in range(1, k + 1)}
    ideriv = mhigh
    for _m in range(2, mhigh + 1):
        jp1mid = 1
        for j in range(ideriv, k + 1):
            d[(j, ideriv)] = col1[jp1mid]
            jp1mid += 1
        ideriv -= 1
        st = _bsplvb(t, x, left, kp1 - ideriv, st)
    for i in range(1, k + 1):
        d[(i, 1)] = col1[i]
    a = {(i, j): (1.0 if i == j else 0.0) for i in range(1, k + 1) for j in range(1, k + 1)}
    for m in range(2, mhigh + 1):
        kp1mm = kp1 - m
        fkp1mm = float(kp1mm)
        il = left
        i = k
        for _ in range(kp1mm):
            factor = fkp1mm / (t[il + kp1mm - 1] - t[il - 1])
            for j in range(1, i + 1):
                a[(i, j)] = (a[(i, j)] - a[(i - 1, j)]) * factor
            il = il - 1
            i -= 1
        for i in range(1, k + 1):
            s = 0.0
            for j in range(max(i, m), k + 1):
                s = a[(j, i)] * d[(j, m)] + s
            d[(i, m)] = s
    return d


def _sgram(t, nk):
    """``sgram.f``: the four bands of the integrated squared second-derivative Gram matrix."""
    i = np.arange(1, nk + 1)
    left = np.maximum(i, 4)
    x0, x1 = t[i - 1], t[i]
    v0 = _bsplvd(t, x0, left, 3)
    v1 = _bsplvd(t, x1, left, 3)
    y1 = {ii: v0[(ii, 3)] for ii in range(1, 5)}
    y2 = {ii: v1[(ii, 3)] - y1[ii] for ii in range(1, 5)}
    wpt = x1 - x0
    bands = [np.zeros(nk) for _ in range(4)]
    for band in range(4):
        idx, val = [], []
        for ii in range(1, 5 - band):
            jj = ii + band
            term = wpt * ((y1[ii] * y1[jj] + (y2[ii] * y1[jj] + y2[jj] * y1[ii]) * 0.5)
                          + y2[ii] * y2[jj] * _THIRD)
            idx.append(left - 4 + ii - 1)
            val.append(term)
        # contributions reach each entry in Fortran loop order: interval first, then ii
        order = np.argsort(np.repeat(np.arange(nk), len(idx)).reshape(nk, -1).T.ravel(), kind="stable")
        np.add.at(bands[band], np.concatenate(idx)[order], np.concatenate(val)[order])
    return bands


def _stxwx(x, z, w, t, nk):
    """``stxwx.f``: ``X'W z`` and the four bands of ``X'WX`` (weights enter squared)."""
    left = np.minimum(np.searchsorted(t[:nk + 1], x, side="right"), nk)
    v = _bsplvd(t, x, left, 1)
    v = [v[(r, 1)] for r in range(1, 5)]
    w2 = w * w
    y = np.zeros(nk)
    hs = [np.zeros(nk) for _ in range(4)]
    # an entry gets its contributions in data order; points sorted by x have non-decreasing
    # ``left``, so the later rows (r = 3, 2, ...) of the 4 x 4 block reach it first
    for r in range(3, -1, -1):
        j = left - 4 + r                  # 0-based coefficient index
        np.add.at(y, j, w2 * z * v[r])
        for band in range(4 - r):
            if band == 0:
                np.add.at(hs[0], j, w2 * (v[r] * v[r]))
            else:
                np.add.at(hs[band], j, w2 * v[r] * v[r + band])
    return y, hs


def _dpbfa_dpbsl(abd, b):
    """LINPACK ``dpbfa`` + ``dpbsl`` for a symmetric band matrix with 3 super-diagonals.

    Returns ``(solution, factor)``.

    ``abd`` is the Fortran (4, n) band array, row 3 (0-based) the diagonal. Scalar loops as in
    LINPACK; ``ddot``/``daxpy`` of length <= 3 accumulate in order."""
    m = 3
    n = abd.shape[1]
    a = abd.tolist()
    for j in range(n):
        s = 0.0
        ik = m
        jk = max(j - m, 0)
        mu = max(m + 1 - j - 1 + 1, 1) - 1          # 0-based row of the first used entry
        for kk in range(mu, m):
            dot = 0.0
            for l in range(kk - mu):
                dot = dot + a[ik + l][jk] * a[mu + l][j]
            tt = a[kk][j] - dot
            tt = tt / a[m][jk]
            a[kk][j] = tt
            s = s + tt * tt
            ik -= 1
            jk += 1
        s = a[m][j] - s
        if s <= 0.0:
            raise np.linalg.LinAlgError(f"smooth.spline: band matrix not positive definite at {j + 1}")
        a[m][j] = math.sqrt(s)
    return _dpbsl(a, b), a


def _dpbsl(a, b):
    """LINPACK ``dpbsl``: solve with the factor ``a`` (list of 4 rows) from ``dpbfa``."""
    m = 3
    n = len(a[0])
    bb = list(map(float, b))
    for k in range(n):
        lm = min(k, m)
        la = m - lm
        lb = k - lm
        dot = 0.0
        for l in range(lm):
            dot = dot + a[la + l][k] * bb[lb + l]
        bb[k] = (bb[k] - dot) / a[m][k]
    for k in range(n - 1, -1, -1):
        lm = min(k, m)
        la = m - lm
        lb = k - lm
        bb[k] = bb[k] / a[m][k]
        tt = -bb[k]
        for l in range(lm):
            bb[lb + l] = bb[lb + l] + tt * a[la + l][k]
    return np.asarray(bb)


def _sinerp_p1ip(a, nk):
    """``sinerp.f`` pass 1: the 4 bands of inner products of the columns of R^-1."""
    p1 = [[0.0] * nk for _ in range(4)]
    w3 = [0.0, 0.0, 0.0]
    w2 = [0.0, 0.0]
    w1 = [0.0]
    for i in range(1, nk + 1):
        j = nk - i + 1                                  # 1-based
        c0 = 1.0 / a[3][j - 1]
        if j <= nk - 3:
            c1 = a[0][j + 2] * c0
            c2 = a[1][j + 1] * c0
            c3 = a[2][j] * c0
        elif j == nk - 2:
            c1 = 0.0
            c2 = a[1][j + 1] * c0
            c3 = a[2][j] * c0
        elif j == nk - 1:
            c1 = 0.0
            c2 = 0.0
            c3 = a[2][j] * c0
        else:
            c1 = c2 = c3 = 0.0
        p1[0][j - 1] = 0.0 - (c1 * w3[0] + c2 * w3[1] + c3 * w3[2])
        p1[1][j - 1] = 0.0 - (c1 * w3[1] + c2 * w2[0] + c3 * w2[1])
        p1[2][j - 1] = 0.0 - (c1 * w3[2] + c2 * w2[1] + c3 * w1[0])
        p1[3][j - 1] = (c0 * c0 + c1 * c1 * w3[0] + 2.0 * c1 * c2 * w3[1]
                        + 2.0 * c1 * c3 * w3[2] + c2 * c2 * w2[0] + 2.0 * c2 * c3 * w2[1]
                        + c3 * c3 * w1[0])
        w3[0] = w2[0]
        w3[1] = w2[1]
        w3[2] = p1[1][j - 1]
        w2[0] = w1[0]
        w2[1] = p1[2][j - 1]
        w1[0] = p1[3][j - 1]
    return np.asarray(p1)


def _sslvrg(x, y, w, ssw, knot, nk, xwy, hs, sg, lam, crit=True, dofoff=0.0, penalt=1.0):
    """``sslvrg.f``: coefficients for ``lambda`` and, with ``crit``, the GCV criterion."""
    abd = np.zeros((4, nk))
    abd[3] = hs[0] + lam * sg[0]
    abd[2, 1:] = hs[1][:-1] + lam * sg[1][:-1]
    abd[1, 2:] = hs[2][:-2] + lam * sg[2][:-2]
    abd[0, 3:] = hs[3][:-3] + lam * sg[3][:-3]
    coef, a = _dpbfa_dpbsl(abd, xwy)
    if not crit:
        return coef, None
    sz = _bvalue(knot, coef, nk, x, 0)
    p1 = _sinerp_p1ip(a, nk)
    # leverages: interval of each point (``interv`` over knot(1..nk+1)), the end point moved inside
    xv = x.copy()
    ileft = np.searchsorted(knot[:nk + 1], xv, side="right")
    right = xv >= knot[nk]
    ileft = np.where(right, nk, ileft)
    xv = np.where(right, knot[nk] - 1e-11, xv)
    left = xv < knot[0]
    ileft = np.where(left, 4, ileft)
    xv = np.where(left, knot[3] + 1e-11, xv)
    v = _bsplvd(knot, xv, ileft, 1)
    b0, b1, b2, b3 = (v[(r, 1)] for r in range(1, 5))
    j = ileft - 4                                       # 0-based column of p1ip(., j)
    lev = (p1[3, j] * (b0 * b0) + 2.0 * p1[2, j] * b0 * b1
           + 2.0 * p1[1, j] * b0 * b2 + 2.0 * p1[0, j] * b0 * b3
           + p1[3, j + 1] * (b1 * b1) + 2.0 * p1[2, j + 1] * b1 * b2
           + 2.0 * p1[1, j + 1] * b1 * b3 + p1[3, j + 2] * (b2 * b2)
           + 2.0 * p1[2, j + 2] * b2 * b3 + p1[3, j + 3] * (b3 * b3)) * (w * w)
    r = (y - sz) * w
    rss = np.cumsum(np.concatenate(([ssw], r * r)))[-1]
    df = np.cumsum(np.concatenate(([0.0], lev)))[-1]
    sumw = np.cumsum(np.concatenate(([0.0], w * w)))[-1]
    t = 1.0 - (dofoff + penalt * df) / sumw
    return coef, (rss / sumw) / (t * t)


def _r_pow16(e):
    """``R_pow(16, e)``: ``powl`` on x86-64, rounded to double."""
    return float(np.power(np.longdouble(16.0), np.longdouble(e)))


def _bvalue(t, coef, nk, x, jderiv=0):
    """``bvalue.f`` (cubic) at points ``x`` inside [t(4), t(nk+1)]."""
    nkk = nk + _K
    i = np.searchsorted(t[:nkk], x, side="right")       # largest 1-based i with t(i) <= x
    at_end = (x == t[nk]) & (t[nk] == t[nkk - 1])
    i = np.where(at_end, nk, i)
    i = np.minimum(i, nk)
    dl = {j: x - t[i + 1 - j - 1] for j in range(1, _K)}
    dr = {j: t[i + j - 1] - x for j in range(1, _K)}
    aj = {jc: coef[i - _K + jc - 1] for jc in range(1, _K + 1)}
    for j in range(1, jderiv + 1):
        kmj = _K - j
        fkmj = float(kmj)
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = ((aj[jj + 1] - aj[jj]) / (dl[ilo] + dr[jj])) * fkmj
            ilo -= 1
    for j in range(jderiv + 1, _K):
        kmj = _K - j
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = (aj[jj + 1] * dl[ilo] + aj[jj] * dr[jj]) / (dl[ilo] + dr[jj])
            ilo -= 1
    return aj[1]


class SmoothSpline:
    """A fitted ``smooth.spline(x, y, spar = spar)`` (no weights), with R's ``predict``.

    ``spar=None`` chooses ``spar`` by generalised cross-validation, as ``smooth.spline(x, y)``."""

    def __init__(self, x, y, spar: float | None = 1.0):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("missing or infinite values in inputs are not allowed")
        n = x.size
        tol = 1e-6 * iqr(x)
        if not (np.isfinite(tol) and tol > 0):
            raise ValueError("'tol' must be strictly positive and finite (IQR(x) is 0)")
        xx = np.rint((x - r_mean(x)) / tol)
        iox = np.argsort(x, kind="stable")
        xxs = xx[iox]
        nd = np.concatenate(([True], xxs[:-1] < xxs[1:]))
        ux = x[iox][nd]
        nx = ux.size
        if nx <= 3:
            raise ValueError("need at least four unique 'x' values")
        if nx == n:
            wbar = np.ones(nx)
            ybar = y[iox]
            yssw = 0.0
        else:
            grp = np.cumsum(nd) - 1                       # group of each sorted point
            g = np.empty(n, dtype=np.int64)
            g[iox] = grp
            wbar = np.bincount(g, minlength=nx).astype(np.float64)
            sy = np.zeros(nx, dtype=np.longdouble)
            np.add.at(sy, g, y.astype(np.longdouble))     # R sum(): long double, original order
            syy = np.zeros(nx, dtype=np.longdouble)
            np.add.at(syy, g, (y * y).astype(np.longdouble))
            ybar = sy.astype(np.float64) / np.where(wbar > 0, wbar, 1.0)
            yssw = float(np.cumsum((syy.astype(np.float64) - wbar * (ybar * ybar)).astype(np.longdouble))[-1])
        r_ux = ux[-1] - ux[0]
        xbar = (ux - ux[0]) / r_ux
        nknots = _nknots(nx)
        sel = _seq_length_out(1.0, float(nx), nknots).astype(np.int64) - 1
        knot = np.concatenate((np.repeat(xbar[0], 3), xbar[sel], np.repeat(xbar[-1], 3)))
        nk = nknots + 2
        ws = np.sqrt(wbar)
        sg = _sgram(knot, nk)
        xwy, hs = _stxwx(xbar, ybar, ws, knot, nk)
        t1 = np.cumsum(hs[0][2:nk - 3])[-1] if nk > 5 else 0.0
        t2 = np.cumsum(sg[0][2:nk - 3])[-1] if nk > 5 else 0.0
        ratio = t1 / t2

        def comp(sp, crit):
            lam_ = ratio * _r_pow16(sp * 6.0 - 2.0)
            c_, f_ = _sslvrg(xbar, ybar, ws, yssw, knot, nk, xwy, hs, sg, lam_, crit=crit)
            return lam_, c_, f_

        if spar is not None:
            lam, coef, _ = comp(float(spar), False)
            spar_out, crit_out = float(spar), None
        else:
            spar_out, lam, coef, crit_out = self._gcv_search(comp)
        self.coef = coef
        self.knot, self.nk = knot, nk
        self.min, self.range = ux[0], r_ux
        self.spar, self.ratio, self.lambda_, self.crit = spar_out, float(ratio), float(lam), crit_out
        self.x, self.w, self.yin = ux, wbar, ybar

    @staticmethod
    def _gcv_search(comp, lspar=-1.5, uspar=1.5, tol=1e-4, eps=2e-8, maxit=500):
        """``sbart.c``: Brent's golden-section / parabolic search for the GCV-optimal ``spar``.

        As in R, the returned coefficients and ``lambda`` are those of the last evaluation."""
        c_gold = 0.381966011250105151795413165634
        big = 1e100

        def fsign(a_, b_):
            return abs(a_) if b_ >= 0 else -abs(a_)

        a, b = lspar, uspar
        v = a + c_gold * (b - a)
        w = x = v
        e = 0.0
        d = 0.0
        lam, coef, fx = comp(x, True)
        fv = fw = fx
        it = 0
        while True:
            xm = (a + b) * 0.5
            tol1 = eps * abs(x) + tol / 3.0
            tol2 = tol1 * 2.0
            it += 1
            if abs(x - xm) <= tol2 - (b - a) * 0.5 or it > maxit:
                break
            golden = abs(e) <= tol1 or fx >= big or fv >= big or fw >= big
            if not golden:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = (q - r) * 2.0
                if q > 0.0:
                    p = -p
                q = abs(q)
                r = e
                e = d
                if abs(p) >= abs(0.5 * q * r) or q == 0.0 or p <= q * (a - x) or p >= q * (b - x):
                    golden = True
                else:
                    d = p / q
                    u = x + d
                    if u - a < tol2 or b - u < tol2:
                        d = fsign(tol1, xm - x)
            if golden:
                e = (a - x) if x >= xm else (b - x)
                d = c_gold * e
            u = x + (d if abs(d) >= tol1 else fsign(tol1, d))
            lam, coef, fu = comp(u, True)
            if not np.isfinite(fu):
                fu = 2.0 * big
            if fu <= fx:
                if u >= x:
                    a = x
                else:
                    b = x
                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, fv = w, fw
                    w, fw = u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
        return x, lam, coef, fx

    def _eval(self, xs, deriv):
        return _bvalue(self.knot, self.coef, self.nk, xs, deriv)

    def predict(self, x, deriv: int = 0):
        """``predict(fit, x)$y`` (R ``predict.smooth.spline.fit``), linear outside the data."""
        x = np.asarray(x, dtype=np.float64)
        xs = (x - self.min) / self.range
        left, right = xs < 0, xs > 1
        inside = ~(left | right)
        y = xs.copy()
        if inside.any():
            y[inside] = self._eval(xs[inside], deriv)
        if (left | right).any():
            xr = np.array([self.min, self.min + self.range])
            xrs = (xr - self.min) / self.range
            if deriv == 0:
                end = self._eval(xrs, 0)
                slope = (self._eval(xrs, 1) / self.range) * self.range
                y[left] = end[0] + slope[0] * (xs[left] - 0)
                y[right] = end[1] + slope[1] * (xs[right] - 1)
            elif deriv == 1:
                slope = (self._eval(xrs, 1) / self.range) * self.range
                y[left], y[right] = slope[0], slope[1]
            else:
                y[left | right] = 0.0
        if deriv > 0:
            y = y / (self.range ** deriv)
        return y

    __call__ = predict


def smooth_spline(x, y, spar: float | None = 1.0) -> SmoothSpline:
    """Fit a :class:`SmoothSpline` (functional alias of the constructor)."""
    return SmoothSpline(x, y, spar=spar)

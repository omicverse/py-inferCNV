"""Small pieces of base R, with R's arithmetic, used by the scCDC port.

``mean`` and ``var`` accumulate in long double with a second correction pass (R's summary.c and
cov.c); ``round(x, 3)`` follows R >= 4.0's ``fround``; ``p.adjust(method = "fdr")`` multiplies by
``n / i`` as R does.
"""
from __future__ import annotations

import math

import numpy as np

_LD = np.longdouble


def _seqsum(v) -> np.longdouble:
    """Sequential long-double sum (R's LDOUBLE loops); ``cumsum`` adds in order."""
    v = np.asarray(v, dtype=_LD)
    return np.cumsum(v)[-1] if v.size else _LD(0)


def r_mean(x) -> float:
    """``mean(x)`` for a double vector."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    s = _seqsum(x) / n
    if np.isfinite(float(s)):
        s = s + _seqsum(x.astype(_LD) - s) / n
    return float(s)


def r_sd(x) -> float:
    """``sd(x)`` = ``sqrt(var(x))`` (cov.c: long-double mean, then sum of products / (n - 1))."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    tmp = _seqsum(x) / n
    if np.isfinite(float(tmp)):
        tmp = tmp + _seqsum(x.astype(_LD) - tmp) / n
    xm = float(tmp)
    d = x - xm
    return float(np.sqrt(float(_seqsum(d * d) / (n - 1))))


def iqr(x) -> float:
    """``IQR(x)``: ``diff(quantile(x, c(.25, .75)))`` with type 7."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    out = []
    for p in (0.25, 0.75):
        index = 1 + max(n - 1, 0) * p
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        qs = x[lo - 1]
        if index > lo and x[hi - 1] != qs:
            h = index - lo
            qs = (1 - h) * qs + h * x[hi - 1]
        out.append(qs)
    return out[1] - out[0]


def round3_index(x) -> np.ndarray:
    """Integer ``k`` with ``round(x, 3) == k / 1000`` (R ``fround``, x >= 0).

    R picks between ``floor(1000 x) / 1000`` and ``ceiling(1000 x) / 1000`` whichever is closer to
    ``x``, the even one on a tie."""
    x = np.asarray(x, dtype=np.float64)
    x10 = 1000.0 * x
    i10 = np.floor(x10)
    xd = i10 / 1000.0
    xu = np.ceil(x10) / 1000.0
    du, dd = xu - x, x - xd
    up = (du < dd) | ((du == dd) & (np.fmod(i10, 2.0) == 1))
    return np.where(up, np.ceil(x10), i10).astype(np.int64)


def p_adjust_fdr(p) -> np.ndarray:
    """``p.adjust(p, method = "fdr")`` (Benjamini-Hochberg) for p without NA."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if n <= 1:
        return p.copy()
    i = np.arange(n, 0, -1, dtype=np.float64)
    o = np.argsort(-p, kind="stable")          # order(p, decreasing = TRUE)
    adj = np.minimum(1.0, np.minimum.accumulate((n / i) * p[o]))
    out = np.empty(n)
    out[o] = adj
    return out


def row_means(m) -> np.ndarray:
    """``rowMeans`` of a dense matrix (long double, columns added in order)."""
    m = np.asarray(m, dtype=np.float64)
    s = np.zeros(m.shape[0], dtype=_LD)
    for j in range(m.shape[1]):
        s += m[:, j]
    return (s / m.shape[1]).astype(np.float64)


def c_apply(fn, x) -> np.ndarray:
    """Apply a C math-library function (``math.log``, ``math.exp``, ...) element-wise.

    NumPy's SIMD ``log``/``exp`` can differ from the C library (which R uses) in the last bit;
    repeated values are evaluated once."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    u, inv = np.unique(x, return_inverse=True)
    out = np.fromiter(map(fn, u.tolist()), dtype=np.float64, count=u.size)
    return out[inv].reshape(x.shape)


_PA = (2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582, 18154.981253343561249,
       0.065682337918207449113)
_PB = (47.20258190468824187, 976.09855173777669322, 10260.932208618978205, 45507.789335026729956)
_PC = (0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979, 597.27027639480026226,
       2494.5375852903726711, 6848.1904505362823326, 11602.651437647350124, 9842.7148383839780218,
       1.0765576773720192317e-8)
_PD = (22.266688044328115691, 235.38790178262499861, 1519.377599407554805, 6485.558298266760755,
       18615.571640885098091, 34900.952721145977266, 38912.003286093271411, 19685.429676859990727)
_PP = (0.21589853405795699, 0.1274011611602473639, 0.022235277870649807, 0.001421619193227893466,
       2.9112874951168792e-5, 0.02307344176494017303)
_PQ = (1.28426009614491121, 0.468238212480865118, 0.0659881378689285515, 0.00378239633202758244,
       7.29751555083966205e-5)
_SQRT_32 = 5.656854249492380195206754896838
_1_SQRT_2PI = 0.398942280401432677939946059934


def pnorm(q, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    """``pnorm(q, mean, sd)`` (lower tail): R's ``pnorm5`` / ``pnorm_both`` (Cody 1993)."""
    x = (np.asarray(q, dtype=np.float64) - mean) / sd
    y = np.abs(x)
    cum = np.where(x > 0, 1.0, 0.0)                       # |x| beyond the tails
    eps = np.finfo(np.float64).eps * 0.5
    # |x| <= qnorm(3/4)
    s = y <= 0.67448975
    if s.any():
        xs = x[s]
        xsq = xs * xs
        xnum = _PA[4] * xsq
        xden = xsq.copy()
        for i in range(3):
            xnum = (xnum + _PA[i]) * xsq
            xden = (xden + _PB[i]) * xsq
        tiny = np.abs(xs) <= eps
        xnum[tiny] = 0.0
        xden[tiny] = 0.0
        cum[s] = 0.5 + xs * (xnum + _PA[3]) / (xden + _PB[3])

    def tail(xv, temp, base):
        xsq = np.ldexp(np.trunc(np.ldexp(base, 4)), -4)
        dl = (base - xsq) * (base + xsq)
        c = c_apply(math.exp, -xsq * np.ldexp(xsq, -1)) * c_apply(math.exp, -np.ldexp(dl, -1)) * temp
        return np.where(xv > 0, 1.0 - c, c)

    # qnorm(3/4) < |x| <= sqrt(32)
    s = (y > 0.67448975) & (y <= _SQRT_32)
    if s.any():
        yy = y[s]
        xnum = _PC[8] * yy
        xden = yy.copy()
        for i in range(7):
            xnum = (xnum + _PC[i]) * yy
            xden = (xden + _PD[i]) * yy
        cum[s] = tail(x[s], (xnum + _PC[7]) / (xden + _PD[7]), yy)
    # sqrt(32) < |x|, inside the range where the lower tail is not 0 or 1
    s = (y > _SQRT_32) & (x > -38.4674) & (x < 8.2924)
    if s.any():
        xs, yy = x[s], y[s]
        xsq = 1.0 / (xs * xs)
        xnum = _PP[5] * xsq
        xden = xsq.copy()
        for i in range(4):
            xnum = (xnum + _PP[i]) * xsq
            xden = (xden + _PQ[i]) * xsq
        temp = xsq * (xnum + _PP[4]) / (xden + _PQ[4])
        temp = (_1_SQRT_2PI - temp) / yy
        cum[s] = tail(xs, temp, xs)
    return cum


def qnorm(p, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    """``qnorm(p, mean, sd)`` (lower tail, not log): R's ``qnorm5`` (Wichura's AS 241).

    The tails use the C library ``log``/``sqrt``; ``p`` must lie in (0, 1)."""
    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    val = np.empty_like(p)
    c = np.abs(q) <= 0.425
    if c.any():
        qq = q[c]
        r = 0.180625 - qq * qq
        val[c] = (qq * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                              + 67265.770927008700853) * r + 45921.953931549871457) * r
                            + 13731.693765509461125) * r + 1971.5909503065514427) * r
                          + 133.14166789178437745) * r + 3.387132872796366608)
                  / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                          + 39307.89580009271061) * r + 21213.794301586595867) * r
                        + 5394.1960214247511077) * r + 687.1870074920579083) * r
                      + 42.313330701600911252) * r + 1.0))
    t = ~c
    if t.any():
        qt, pt = q[t], p[t]
        lp = c_apply(math.log, np.where(qt > 0, 1.0 - pt, pt))
        r = c_apply(math.sqrt, -lp)
        out = np.empty_like(r)
        a = r <= 5.0
        if a.any():
            ra = r[a] - 1.6
            out[a] = ((((((((ra * 7.7454501427834140764e-4 + 0.0227238449892691845833) * ra
                            + 0.24178072517745061177) * ra + 1.27045825245236838258) * ra
                          + 3.64784832476320460504) * ra + 5.7694972214606914055) * ra
                        + 4.6303378461565452959) * ra + 1.42343711074968357734)
                      / (((((((ra * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * ra
                              + 0.0151986665636164571966) * ra + 0.14810397642748007459) * ra
                            + 0.68976733498510000455) * ra + 1.6763848301838038494) * ra
                          + 2.05319162663775882187) * ra + 1.0))
        b = (~a) & (r <= 27.0)
        if b.any():
            rb = r[b] - 5.0
            out[b] = ((((((((rb * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * rb
                            + 0.0012426609473880784386) * rb + 0.026532189526576123093) * rb
                          + 0.29656057182850489123) * rb + 1.7848265399172913358) * rb
                        + 5.4637849111641143699) * rb + 6.6579046435011037772)
                      / (((((((rb * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * rb
                              + 1.8463183175100546818e-5) * rb + 7.868691311456132591e-4) * rb
                            + 0.0148753612908506148525) * rb + 0.13692988092273580531) * rb
                          + 0.59983220655588793769) * rb + 1.0))
        if (r > 27.0).any():
            raise ValueError("qnorm: p closer to 0 or 1 than exp(-729) is not supported")
        val[t] = np.where(qt < 0.0, -out, out)
    return mean + sd * val

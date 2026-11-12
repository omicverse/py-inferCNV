"""The Seurat 5 preprocessing that DoubletFinder runs on every merged real + artificial matrix.

``doubletFinder`` and ``paramSweep`` call ``NormalizeData`` (LogNormalize), ``FindVariableFeatures``
(vst, 2,000 genes), ``ScaleData`` and ``RunPCA`` with their defaults. This module ports those steps:

* ``r_loess`` — R ``stats::loess`` for one predictor with its default ``surface = "interpolate"``
  (loessf.f: k-d tree with at most ``floor(n * span * cell)`` points per cell, local tricube-weighted
  regressions at the cell vertices, cubic Hermite interpolation between them).
* ``vst`` — Seurat ``VST``: loess of log10 variance on log10 mean, then the variance of the
  standardised counts, clipped above at ``sqrt(ncol)``.
* ``scale_data`` — Seurat ``FastSparseRowScale``: centre, divide by the sd, clip above at 10.
* ``run_pca`` — the cell embeddings ``u * d`` of the scaled matrix (no further centring).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

_EPS = np.finfo(float).eps


def _local_fit(x, y, q, nf, degree):
    """loessf.f ``ehg127``: value and slope at ``q`` of the tricube-weighted local polynomial."""
    dist = (x - q) ** 2
    sel = np.argpartition(dist, nf - 1)[:nf]
    rho = dist[sel].max()
    w = np.sqrt((1.0 - np.sqrt(dist[sel] / rho) ** 3) ** 3)
    dx = x[sel] - q
    cols = [w, w * dx] + ([w * dx ** 2] if degree >= 2 else [])
    b = np.column_stack(cols)
    eta = w * y[sel]
    colnor = np.sqrt((b ** 2).sum(axis=0))
    colnor[colnor == 0] = 1.0
    b = b / colnor
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    tol = s[0] * (100 * _EPS)
    g = np.where(s > tol, (u.T @ eta) / np.where(s > tol, s, 1.0), 0.0)
    coef = (vt.T @ g) / colnor
    return coef[0], coef[1]


def r_loess(x, y, span: float = 0.3, degree: int = 2, cell: float = 0.2) -> np.ndarray:
    """Fitted values of R ``loess(y ~ x, span, degree)`` with the default interpolated surface."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    nf = min(n, int(np.floor(n * span + 1e-5)))
    fc = int(np.floor(n * (span * cell)))
    nvmax = max(200, n)
    xs = np.sort(x)                                              # xs[i - 1] is the i-th smallest
    lo_, hi_ = xs[0], xs[-1]
    mu = 0.005 * max(hi_ - lo_, 1e-10 * max(abs(lo_), abs(hi_)) + 1e-30)
    verts = [lo_ - mu, hi_ + mu]
    cells = [(1, n, 0, 1)]                                       # (l, u, lower vertex, upper vertex)
    leaves, splits = [], []
    p = 0
    while p < len(cells):                                        # ehg124: cells in creation order
        l, u, a, b = cells[p]
        p += 1
        leaf = (u - l + 1) <= fc or verts[b] - verts[a] <= 0.0
        if not leaf:
            leaf = nvmax < len(cells) + 2 or nvmax < len(verts) + 1
        if not leaf:
            m = (l + u) // 2
            off = 0
            while l <= m + off < u:                              # ties all go to the upper son
                if xs[m + off - 1] == xs[m + off]:
                    off = -off
                    if off >= 0:
                        off += 1
                else:
                    m += off
                    break
            t = xs[m - 1]
            leaf = verts[a] == t or verts[b] == t
        if leaf:
            leaves.append((a, b))
            continue
        try:
            v = verts.index(t)
        except ValueError:
            verts.append(t)
            v = len(verts) - 1
        splits.append(t)
        cells.append((l, m, a, v))
        cells.append((m + 1, u, v, b))
    vval = np.array([_local_fit(x, y, q, nf, degree) for q in verts])     # ehg139
    vx = np.asarray(verts)
    order = np.argsort([vx[a] for a, _ in leaves])
    la = np.array([leaves[i][0] for i in order])
    lb = np.array([leaves[i][1] for i in order])
    leaf = np.searchsorted(np.sort(splits), x, side="left")              # z <= xi goes left
    a, b = la[leaf], lb[leaf]
    width = vx[b] - vx[a]
    h = (x - vx[a]) / width
    phi0, phi1 = (1 - h) ** 2 * (1 + 2 * h), h ** 2 * (3 - 2 * h)
    psi0, psi1 = h * (1 - h) ** 2, h ** 2 * (h - 1)
    return phi0 * vval[a, 0] + phi1 * vval[b, 0] + (psi0 * vval[a, 1] + psi1 * vval[b, 1]) * width


def _row_mean_var(mat):
    """Row means and (n - 1) variances of a genes x cells matrix."""
    n = mat.shape[1]
    if sparse.issparse(mat):
        mat = sparse.csr_matrix(mat)
        mean = np.asarray(mat.sum(axis=1)).ravel() / n
        sq = np.asarray(mat.multiply(mat).sum(axis=1)).ravel()
    else:
        mat = np.asarray(mat, dtype=np.float64)
        mean = mat.mean(axis=1)
        sq = (mat ** 2).sum(axis=1)
    var = (sq - n * mean ** 2) / (n - 1)
    return mean, np.maximum(var, 0.0)


def vst(counts, n_features: int = 2000, span: float = 0.3) -> np.ndarray:
    """Seurat ``VST`` on a genes x cells count matrix; returns the selected rows in rank order."""
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    n = counts.shape[1]
    mean, _ = _row_mean_var(counts)
    # Seurat SparseRowVar2: sum over nonzeros of (x - mu)^2 plus the zeros' mu^2
    nnz = np.diff(counts.indptr)
    rows = np.repeat(np.arange(counts.shape[0]), nnz)
    dev = np.bincount(rows, (counts.data - mean[rows]) ** 2, minlength=counts.shape[0])
    var = (dev + (n - nnz) * mean ** 2) / (n - 1)
    keep = var > 0
    expected = np.zeros_like(var)
    expected[keep] = 10.0 ** r_loess(np.log10(mean[keep]), np.log10(var[keep]), span=span)
    sd = np.sqrt(expected)
    vmax = np.sqrt(n)
    ok = sd > 0
    z = np.zeros_like(counts.data)
    sdr = sd[rows]
    nzok = sdr > 0
    z[nzok] = np.minimum((counts.data[nzok] - mean[rows][nzok]) / sdr[nzok], vmax)
    std_sum = np.bincount(rows, z ** 2, minlength=counts.shape[0])
    std_sum[ok] += (n - nnz[ok]) * (mean[ok] / sd[ok]) ** 2
    vstd = np.where(ok, std_sum / (n - 1), 0.0)
    return np.argsort(-vstd, kind="stable")[:n_features]


def log_normalize(counts, scale_factor: float = 1e4):
    """Seurat ``LogNormalize``: ``log1p(x / colSums * scale_factor)``, genes x cells, sparse in/out."""
    counts = sparse.csc_matrix(counts, dtype=np.float64)
    lib = np.asarray(counts.sum(axis=0)).ravel()
    out = counts.copy()
    out.data = np.log1p(out.data / np.repeat(lib, np.diff(out.indptr)) * scale_factor)
    return out


def scale_data(mat, scale_max: float = 10.0) -> np.ndarray:
    """Seurat ``FastSparseRowScale`` on a genes x cells matrix: per gene, the mean and the sum of
    squared deviations accumulated over the stored entries in cell order (the zeros' share added
    as ``mean^2 * n_zero``), ``sd = sqrt(ss / (n - 1))``, ``(x - mean) / sd`` clipped above."""
    x = mat.toarray() if sparse.issparse(mat) else np.array(mat, dtype=np.float64)
    n = x.shape[1]
    nz = x != 0
    mean = np.cumsum(np.where(nz, x, 0.0), axis=1)[:, -1] / n
    d = np.where(nz, x - mean[:, None], 0.0)
    ss = np.cumsum(d * d, axis=1)[:, -1] + mean * mean * (n - nz.sum(axis=1))
    sd = np.sqrt(ss / (n - 1))
    out = (x - mean[:, None]) / sd[:, None]
    np.minimum(out, scale_max, out=out)
    return out


def run_pca(scaled: np.ndarray, n_pcs: int = 10) -> np.ndarray:
    """Cell embeddings ``u * d`` of the scaled features x cells matrix (Seurat ``RunPCA``)."""
    from scipy.linalg import eigh

    a = np.asarray(scaled, dtype=np.float64).T                     # cells x features
    n, p = a.shape
    k = min(n_pcs, n, p)
    # only the top k eigenpairs, of the smaller Gram matrix
    if n <= p:
        evals, evecs = eigh(a @ a.T, subset_by_index=[n - k, n - 1])
        order = np.argsort(evals)[::-1]
        return evecs[:, order] * np.sqrt(np.maximum(evals[order], 0.0))
    evals, evecs = eigh(a.T @ a, subset_by_index=[p - k, p - 1])
    return a @ evecs[:, np.argsort(evals)[::-1]]


def preprocess(counts, n_pcs: int = 10, n_features: int = 2000) -> np.ndarray:
    """NormalizeData -> FindVariableFeatures(vst) -> ScaleData -> RunPCA on a genes x cells matrix."""
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    feat = vst(counts, n_features=n_features)
    data = log_normalize(counts)[feat]
    return run_pca(scale_data(data), n_pcs=n_pcs)

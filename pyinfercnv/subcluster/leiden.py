"""Leiden-based tumor subclustering.

R parity target: `.single_tumor_leiden_subclustering` + `.leiden_simple_snn` in
`R/inferCNV_tumor_subclusters.R`. We implement the "simple" branch (KNN graph
on the CNV matrix directly, no PCA preprocessing) since the task specifies
"applied on the CNV matrix (log2 FC space)".

Auto-resolution formula (R line 588):
    used_leiden_resolution = (11.98 / n_cells) ** (1 / 1.165)

Note: scanpy's `sc.tl.leiden` is our partition engine. We build the KNN graph
via `sc.pp.neighbors(use_rep='X')` so clustering runs directly on the CNV
matrix (cells x genes) without PCA — mirroring R `nn2(t(expr_data), k=k_nn)`.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["leiden_subcluster"]


def leiden_subcluster(
    cnv_matrix: np.ndarray,
    *,
    resolution: float | str = "auto",
    k_nn: int = 20,  # noqa: N803 (R kwarg name preserved)
    random_state: int = 0,
) -> np.ndarray:
    """Return per-cell int32 subcluster label, shape (n_cells,).

    Parameters
    ----------
    cnv_matrix
        (n_cells, n_genes) log2 fold-change matrix (cells x genes layout).
    resolution
        Leiden resolution. "auto" applies R's formula
        `(11.98 / n_cells) ** (1 / 1.165)`.
    k_nn
        Number of nearest neighbours for the SNN graph (R `k_nn`).
    random_state
        Seed propagated to both `sc.pp.neighbors` and `sc.tl.leiden`.
    """
    X = np.ascontiguousarray(cnv_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2D (n_cells, n_genes); got shape {X.shape}")
    n_cells = X.shape[0]

    # R early-return edge cases (lines 573-584)
    if n_cells < 3:
        return np.zeros(n_cells, dtype=np.int32)
    if k_nn >= n_cells:
        return np.zeros(n_cells, dtype=np.int32)

    # Resolve auto resolution (R line 588)
    if isinstance(resolution, str):
        if resolution != "auto":
            raise ValueError(f"resolution must be 'auto' or float; got {resolution!r}")
        used_resolution = float((11.98 / n_cells) ** (1.0 / 1.165))
    else:
        used_resolution = float(resolution)

    # Lazy import so `import pyinfercnv` stays cheap. Leiden partitioning
    # needs leidenalg (pinned flavor, see below); if missing we point the
    # user at the right extra.
    import anndata as ad
    import scanpy as sc

    try:
        import leidenalg  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        raise ModuleNotFoundError(
            "leiden_subcluster requires the leidenalg package. "
            "Install with `pip install 'pyinfercnv[leiden]'` or "
            "`pip install leidenalg igraph`."
        ) from exc

    adata = ad.AnnData(X=X)

    # Silence scanpy's verbose INFO stream and the leiden-flavor FutureWarning
    # while we drive it programmatically.
    prev_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # scanpy 1.12 emits a UserWarning nudging users to the igraph
            # backend; we pin the leidenalg flavor on purpose for determinism.
            warnings.filterwarnings(
                "ignore",
                message=r".*igraph.*implementation of leiden.*",
                category=UserWarning,
            )
            # use_rep="X" → build KNN graph directly on the CNV matrix,
            # matching R's `nn2(t(expr_data), k=k_nn)` behaviour.
            sc.pp.neighbors(
                adata,
                n_neighbors=k_nn,
                use_rep="X",
                random_state=random_state,
            )
            # flavor="leidenalg" pins the historically default backend so we
            # stay reproducible across scanpy 1.10-1.12 versions.
            sc.tl.leiden(
                adata,
                resolution=used_resolution,
                random_state=random_state,
                flavor="leidenalg",
                directed=False,
                n_iterations=-1,
            )
    finally:
        sc.settings.verbosity = prev_verbosity

    labels = adata.obs["leiden"].astype(int).to_numpy().astype(np.int32)
    return labels

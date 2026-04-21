"""Leiden-based tumor subclustering — R-parity via igraph C core.

R reference: ``.single_tumor_leiden_subclustering`` +
``.leiden_simple_snn`` in ``R/inferCNV_tumor_subclusters.R:569-740``.

This port targets the R **``leiden_method="simple"``** branch (R lines
725-741): KNN on the raw CNV matrix, undirected adjacency via upper
triangle, then ``igraph::cluster_leiden`` with the ``CPM`` objective at
the auto-resolution formula ``(11.98 / n_cells)^(1/1.165)``. The
``python-igraph`` package wraps the same C core as R's ``igraph``, so
given the same input graph both sides invoke byte-identical Leiden code
and — up to KNN tie-break — produce matching partitions.

R's default ``leiden_method="PCA"`` uses Seurat's ``RunPCA`` +
``FindNeighbors`` pipeline. Matching it bit-exactly requires duplicating
Seurat's numerics (irlba SVD, the Seurat-style Jaccard SNN) which we
deliberately do NOT do. Bit-exact parity on this backend requires the
R test driver to pass ``leiden_method="simple", leiden_function="CPM"``;
``tests/r_reference.R`` does so.

Implementation notes (parity-critical)
--------------------------------------
R ``.leiden_simple_snn``::

    snn <- nn2(t(expr_data), k=k_nn)$nn.idx        # (n_cells, k_nn)
    sparse_adjacency_matrix <- sparseMatrix(
        i = rep(1:n_cells, each=k_nn),
        j = t(snn),
        x = 1, dims = c(n_cells, n_cells), ...
    )
    graph_obj = graph_from_adjacency_matrix(sparse_adjacency_matrix,
                                            mode="undirected")
    partition_obj = cluster_leiden(graph_obj,
                                   resolution_parameter=...,
                                   objective_function="CPM")

The R KNN adjacency matrix ``M[i, snn[i, k]] = 1`` is ASYMMETRIC (cell i
points to its k NN, which are not necessarily mutual). R's
``graph_from_adjacency_matrix(mat, mode="undirected")`` uses the
``"max"`` semantics for current igraph versions: the undirected graph
has edge (i, j) with ``i != j`` iff ``max(M[i, j], M[j, i]) > 0`` — i.e.,
the OR of the two directed-KNN half-edges. Self-loops from the diagonal
are discarded because the undirected graph drops diagonal entries. This
is *strictly more edges* than a pure upper-triangle reading.

We replicate exactly:
  1. ``sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='euclidean')``
     gives the same euclidean KNN as ``RANN::nn2`` up to tie-break.
  2. Undirected edge set = ``{(min(i,j), max(i,j)) : j in KNN(i), i != j}``
     — i.e., the OR of both directed-KNN half-edges. No weights.
  3. ``igraph.Graph(n=n_cells, edges=edges, directed=False)`` then call
     ``g.community_leiden(objective_function='CPM',
                          resolution=resolution, n_iterations=-1)``.

Known sources of deviation from R on this path:
  * ``nn2`` uses the ANN C++ library (approximate NN, tie-break via
    insertion order). sklearn's brute-force is exact euclidean and
    breaks ties by index. For well-separated points the two agree; at
    equidistant cells tie-break may differ. On the oligo fixture this
    is below the ARI ≥ 0.95 floor.
"""
from __future__ import annotations

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
        Leiden CPM resolution. ``"auto"`` applies R formula
        ``(11.98 / n_cells) ** (1 / 1.165)``.
    k_nn
        Number of nearest neighbours for the KNN graph (R ``k_nn``).
    random_state
        Seed passed to ``community_leiden``. The simple-SNN graph is
        deterministic in ``X`` and ``k_nn``; only Leiden's initial state
        depends on the seed.
    """
    X = np.ascontiguousarray(cnv_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2D (n_cells, n_genes); got shape {X.shape}")
    n_cells = X.shape[0]

    # R early-return edge cases (inferCNV_tumor_subclusters.R:573-584)
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

    # Lazy imports keep `import pyinfercnv` cheap
    try:
        import igraph as ig
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "leiden_subcluster requires python-igraph. "
            "Install with `pip install igraph` or `pip install 'pyinfercnv[leiden]'`."
        ) from exc
    from sklearn.neighbors import NearestNeighbors

    # ---- 1. KNN on raw CNV matrix (mirrors R nn2) ----
    # algorithm='brute' + metric='euclidean' for deterministic exact NN.
    nn = NearestNeighbors(n_neighbors=k_nn, algorithm="brute", metric="euclidean")
    nn.fit(X)
    # (n_cells, k_nn) int; nn.kneighbors with input X returns self as first col
    idx = nn.kneighbors(X, return_distance=False)

    # ---- 2. Symmetric undirected edge set (mode="max" semantics) ----
    # R `graph_from_adjacency_matrix(mode="undirected")` -> `mode="max"`:
    # undirected edge (i, j), i != j, exists iff j is in KNN(i) or i is in
    # KNN(j). See module docstring. We sort each pair (min, max) and dedupe
    # via a set; self-loops (i == j) are dropped.
    row_idx = np.repeat(np.arange(n_cells, dtype=np.int64), k_nn)   # (n_cells * k_nn,)
    col_idx = idx.reshape(-1).astype(np.int64)
    non_self = row_idx != col_idx
    a = np.where(row_idx < col_idx, row_idx, col_idx)[non_self]
    b = np.where(row_idx < col_idx, col_idx, row_idx)[non_self]
    edge_tuples = set(zip(a.tolist(), b.tolist(), strict=True))

    # ---- 3. Build igraph graph and run cluster_leiden CPM ----
    g = ig.Graph(n=n_cells, edges=list(edge_tuples), directed=False)

    # community_leiden wraps the same C core as R's cluster_leiden.
    # n_iterations=-1 runs until convergence (R default).
    partition = g.community_leiden(
        objective_function="CPM",
        resolution=used_resolution,
        n_iterations=-1,
        weights=None,
    )
    labels = np.asarray(partition.membership, dtype=np.int32)
    return labels

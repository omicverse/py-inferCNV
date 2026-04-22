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

We match the R path's **off-diagonal** semantics; the two graphs are
not guaranteed to be vertex-for-vertex identical (see known deviations
below):
  1. ``sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='euclidean')``
     gives the same euclidean KNN as ``RANN::nn2`` up to tie-break.
     Track A2 empirically verified ``edge Jaccard = 1.0000`` against R
     on three 3CA patients.
  2. Undirected edge set = ``{(min(i,j), max(i,j)) : j in KNN(i), i != j}``
     — the OR of both directed-KNN half-edges. No weights.
     **Self-loops omitted**. R ``nn2`` returns self as the first neighbour
     and R's ``graph_from_adjacency_matrix(mode="undirected")`` keeps the
     diagonal as self-loops. Empirically (brute-forced 967 small random
     graphs in the codex review, see
     ``docs/superpowers/reviews/phase2-i3-fix-codex.md`` S1, and the
     Track A3 sweep in ``benchmarks/phase2/.../track_a1/``) those
     self-loops do not change Leiden membership at the CPM resolution
     infercnv uses.
  3. ``igraph.Graph(n=n_cells, edges=edges, directed=False)`` then call
     ``g.community_leiden(objective_function='CPM',
                          resolution=resolution, n_iterations=-1)``.
     Seeded via ``ig.set_random_number_generator(random.Random(random_state))``
     with a finally-block restore to the default PCG32 so the global
     RNG state does not leak into other callers.

Known sources of deviation from R on this path:
  * ``nn2`` uses the ANN C++ library (approximate NN, tie-break via
    insertion order). sklearn's brute-force is exact euclidean and
    breaks ties by index. For well-separated points the two agree; at
    equidistant cells tie-break may differ.
  * Self-loops omitted (see above).
  * Different PRNG backends under the hood: R igraph uses the bundled
    C PCG32; python-igraph with a ``random.Random`` override dispatches
    through Python's Mersenne-Twister. Leiden's internal seed-usage
    should be limited to node shuffling so the impact is bounded, but
    this is the mechanism why two cross-language runs at the same
    ``random_state`` are not bit-identical.

Track A summary (HANDOFF v6 closeout; see
``benchmarks/phase2/Gao2021_Breast/track_a_summary.md``):
  * DCIS1 (n=1101 tumor): py ARI 0.468±0.14 / R self 0.474±0.12
  * TNBC1 (n=796  tumor): py ARI 0.733±0.09 / R self 0.669±0.10
  * TNBC3 (n=195  tumor): py ARI 0.915±0.04 / R self 0.766±0.29
  Py finds equal-or-higher CPM objective on the same graph; no
  systematic gap observed. Per-seed bit-identity across R/py is
  not a contract.

Optional non-default modes
--------------------------
Two kwargs enable downstream-stability modes (default off):
  * ``n_seeds`` — number of seeds to sweep; returns the partition with
    the highest CPM objective (``rbest``). Default ``1``. Each extra
    seed reuses the same KNN graph, so cost is linear in ``n_seeds``
    for Leiden only.
  * ``min_subcluster_size`` — merge clusters smaller than this into
    their nearest non-small cluster via KNN neighbour voting. Default
    ``None`` (off). Non-CPM-optimal by design: trades a small CPM loss
    for downstream HMM stability by suppressing singletons.
"""
from __future__ import annotations

import numpy as np

__all__ = ["leiden_subcluster"]


def _manual_cpm(
    n: int,
    edges_arr: np.ndarray,
    membership: np.ndarray,
    gamma: float,
) -> float:
    """Σ_c [e_c − γ n_c(n_c−1)/2] on undirected, no-self-loop edge set."""
    m = membership
    K = int(m.max()) + 1 if m.size else 0
    if K == 0:
        return 0.0
    same = m[edges_arr[:, 0]] == m[edges_arr[:, 1]]
    e_c = np.bincount(m[edges_arr[:, 0]][same], minlength=K).astype(np.float64)
    n_c = np.bincount(m, minlength=K).astype(np.float64)
    return float((e_c - gamma * n_c * (n_c - 1) / 2.0).sum())


def _collapse_small_clusters(
    labels: np.ndarray,
    knn_idx: np.ndarray,
    min_size: int,
    max_iter: int = 50,
) -> np.ndarray:
    """Merge clusters of size < ``min_size`` into nearest-neighbour
    non-small cluster by KNN vote. Iterates until convergence or
    bounded max_iter (in case cascading merges are needed)."""
    if min_size is None or min_size <= 1:
        return labels
    labels = labels.astype(np.int32, copy=True)
    n = labels.size
    for _ in range(max_iter):
        sizes = np.bincount(labels)
        small = set(np.where(sizes < min_size)[0].tolist())
        if not small:
            break
        if len(small) >= len(sizes):
            # Entire cell set is "small" — fall back to single cluster.
            labels[:] = 0
            break
        changed = False
        # For each cell in a small cluster, vote its KNN neighbours'
        # non-small cluster labels; reassign to majority (ties → lowest id).
        for i in range(n):
            if labels[i] not in small:
                continue
            # Skip self at col 0, consider remaining KNN neighbours.
            nbr_labels = labels[knn_idx[i, 1:]]
            # Filter out neighbours still in small clusters.
            keep = np.array([lbl not in small for lbl in nbr_labels])
            if not keep.any():
                continue  # leave for next pass (cascade)
            votes = nbr_labels[keep]
            # Majority; ties → lowest label.
            uniq, counts = np.unique(votes, return_counts=True)
            best = uniq[np.argmax(counts[::-1])]  # stable argmax, lowest on tie
            # Better: explicit tie break to lowest id.
            top = counts.max()
            candidates = uniq[counts == top]
            best = int(candidates.min())
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed:
            # Nobody moved this pass; remaining small-cluster cells have
            # no valid non-small neighbour. Dump them into the single
            # largest cluster for determinism.
            sizes = np.bincount(labels)
            non_small = np.array([s >= min_size for s in sizes])
            if not non_small.any():
                labels[:] = 0
                break
            # Largest non-small cluster
            masked_sizes = np.where(non_small, sizes, -1)
            largest = int(np.argmax(masked_sizes))
            for i in range(n):
                if labels[i] in small:
                    labels[i] = largest
            break
    # Compact label space to 0..K-1
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


def leiden_subcluster(
    cnv_matrix: np.ndarray,
    *,
    resolution: float | str = "auto",
    k_nn: int = 20,  # noqa: N803 (R kwarg name preserved)
    random_state: int = 0,
    n_seeds: int = 1,
    min_subcluster_size: int | None = None,
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
    n_seeds
        If > 1, run Leiden with seeds ``random_state, random_state+1,
        …, random_state+n_seeds-1`` and return the partition with the
        highest manual CPM objective (``rbest``). KNN is computed once.
        Default 1 (R-parity; single seed).
    min_subcluster_size
        If set, merge any cluster smaller than this size into its
        nearest non-small cluster via KNN majority vote. Default
        ``None`` (off). **Non-CPM-optimal by design** — trades a small
        CPM loss for downstream stability.
    """
    X = np.ascontiguousarray(cnv_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2D (n_cells, n_genes); got shape {X.shape}")
    n_cells = X.shape[0]

    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}")
    if min_subcluster_size is not None and min_subcluster_size < 1:
        raise ValueError(
            f"min_subcluster_size must be >= 1 or None; got {min_subcluster_size}"
        )

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
    nn = NearestNeighbors(n_neighbors=k_nn, algorithm="brute", metric="euclidean")
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)

    # ---- 2. Symmetric undirected edge set (mode="max" semantics) ----
    row_idx = np.repeat(np.arange(n_cells, dtype=np.int64), k_nn)
    col_idx = idx.reshape(-1).astype(np.int64)
    non_self = row_idx != col_idx
    a = np.where(row_idx < col_idx, row_idx, col_idx)[non_self]
    b = np.where(row_idx < col_idx, col_idx, row_idx)[non_self]
    edge_keys = np.unique(a * n_cells + b)
    edges_arr = np.empty((edge_keys.size, 2), dtype=np.int64)
    edges_arr[:, 0] = edge_keys // n_cells
    edges_arr[:, 1] = edge_keys % n_cells

    # ---- 3. Build igraph graph (reused across seeds if n_seeds > 1) ----
    g = ig.Graph(n=n_cells, edges=edges_arr.tolist(), directed=False)

    def _run_one(seed: int) -> np.ndarray:
        import random as _random
        ig.set_random_number_generator(_random.Random(seed))
        try:
            part = g.community_leiden(
                objective_function="CPM",
                resolution=used_resolution,
                n_iterations=-1,
                weights=None,
            )
        finally:
            ig.set_random_number_generator(None)
        return np.asarray(part.membership, dtype=np.int32)

    if n_seeds == 1:
        labels = _run_one(random_state)
    else:
        # rbest: sweep seeds, pick highest CPM. Tie-break lowest seed offset.
        best_q = -np.inf
        best_lbl: np.ndarray | None = None
        for k in range(n_seeds):
            lbl = _run_one(random_state + k)
            q = _manual_cpm(n_cells, edges_arr, lbl.astype(np.int64), used_resolution)
            if q > best_q:
                best_q = q
                best_lbl = lbl
        assert best_lbl is not None
        labels = best_lbl

    if min_subcluster_size is not None and min_subcluster_size > 1:
        labels = _collapse_small_clusters(labels, idx, int(min_subcluster_size))

    return labels

"""filterHighPNormals — Phase 3 step 19 (R `inferCNV_ops.R:1405-1454`).

Consumes the BayesNet posterior ``cnv_posterior[r, k]`` (from
:func:`pyinfercnv.bayesnet.gibbs.run_bayesnet_gibbs`) and overrides the
HMM state matrix at ``(Genes, Cells)`` positions of any region whose
posterior P(normal) exceeds ``BayesMaxPNormal``, rewriting them to the
neutral state. Mirrors R ``removeCNV`` at ``inferCNV_BayesNet.R:562-630``.

R-parity scope
--------------
* Implements the ``removeCNV`` branch only (R default
  ``postMcmcMethod='removeCNV'``). Every region where
  ``cnv_posterior[r, neutral] > BayesMaxPNormal`` is "removed" by
  overwriting its ``(Genes, Cells)`` block to the neutral state index.
* Does **not** implement the ``reassignCNV`` branch
  (``inferCNV_BayesNet.R:491-540``). R's default ``reassignCNVs=TRUE``
  triggers it after ``removeCNV``, but the Python port defers it to a
  follow-up session; ``run_bayesnet_gibbs`` raises if asked to reassign.

Return contract
---------------
Returns a **new** ``hmm_state_matrix`` (int8, same shape as input) with
overrides applied; the input is not mutated. The merge agent wires this
into :class:`InferCNVResult.bayes_posterior` and into a filtered HMM
state field via ``pipeline_phase3``.
"""
from __future__ import annotations

import numpy as np


# R-parity constants: R uses 1-based state indices (neutral=3 for i6,
# neutral=2 for i3). Python uses 0-based (neutral=2 for i6, neutral=1 for i3).
_NEUTRAL_IDX = {"i6": 2, "i3": 1}


def filter_high_p_normals(
    hmm_state_matrix: np.ndarray,
    bayesnet_result: dict,
    *,
    HMM_type: str = "i6",  # noqa: N803
    BayesMaxPNormal: float = 0.5,  # noqa: N803
) -> np.ndarray:
    """Overwrite regions with posterior P(normal) > threshold back to neutral.

    Parameters
    ----------
    hmm_state_matrix
        Phase 2 HMM state matrix, ``(n_cells, n_bins)`` int8. R 1-based
        indices are already converted to 0-based by
        :func:`pyinfercnv.pipeline_phase2._run_hmm_by_subcluster`.
    bayesnet_result
        Output dict from :func:`pyinfercnv.bayesnet.gibbs.run_bayesnet_gibbs`.
        Consumed fields: ``cnv_posterior`` (n_regions, K),
        ``region_cell_idx`` and ``region_gene_idx`` lists.
    HMM_type
        ``"i6"`` or ``"i3"``. Neutral state index derived per
        :data:`_NEUTRAL_IDX`.
    BayesMaxPNormal
        R ``inferCNV_ops.R:275``. Regions with
        ``cnv_posterior[r, neutral] > BayesMaxPNormal`` are removed
        (overwritten to neutral).

    Returns
    -------
    filtered
        ``(n_cells, n_bins)`` int8, a **new** matrix with the overrides
        applied.
    """
    if HMM_type not in _NEUTRAL_IDX:
        raise ValueError(f"HMM_type must be 'i6' or 'i3', got {HMM_type!r}")
    neutral_idx = _NEUTRAL_IDX[HMM_type]

    cnv_posterior = np.asarray(bayesnet_result["cnv_posterior"], dtype=np.float64)
    region_cell_idx = bayesnet_result["region_cell_idx"]
    region_gene_idx = bayesnet_result["region_gene_idx"]

    n_regions = cnv_posterior.shape[0]
    if n_regions != len(region_cell_idx) or n_regions != len(region_gene_idx):
        raise ValueError(
            "bayesnet_result region arrays length mismatch: "
            f"n_regions={n_regions} cells={len(region_cell_idx)} "
            f"genes={len(region_gene_idx)}"
        )

    filtered = np.array(hmm_state_matrix, dtype=np.int8, copy=True)

    if n_regions == 0:
        return filtered

    p_normal = cnv_posterior[:, neutral_idx]
    to_remove = np.where(p_normal > BayesMaxPNormal)[0]

    for r in to_remove:
        cell_idx = region_cell_idx[r]
        gene_idx = region_gene_idx[r]
        if cell_idx.size == 0 or gene_idx.size == 0:
            continue
        filtered[np.ix_(cell_idx, gene_idx)] = np.int8(neutral_idx)

    return filtered


__all__ = ["filter_high_p_normals"]

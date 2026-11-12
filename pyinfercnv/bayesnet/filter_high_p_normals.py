"""filterHighPNormals — Phase 3 step 19 (R `inferCNV_ops.R:1405-1454`).

Consumes the BayesNet posterior ``cnv_posterior[r, k]`` (from
:func:`pyinfercnv.bayesnet.gibbs.run_bayesnet_gibbs`) and overrides the
HMM state matrix at ``(Genes, Cells)`` positions of any region whose
posterior P(normal) exceeds ``BayesMaxPNormal``, rewriting them to the
neutral state. Mirrors R ``removeCNV`` at ``inferCNV_BayesNet.R:562-630``.

R-parity scope
--------------
Removes high-P(normal) regions and optionally reassigns retained regions to
posterior-argmax states. The matrix and exported table use zero-based states;
the table also includes an explicit one-based state_r for R interoperability.

Return contract
---------------
Returns a **new** ``hmm_state_matrix`` (int8, same shape as input) with
overrides applied; the input is not mutated. ``pipeline_phase3`` stores
the filtered state matrix alongside the unfiltered posterior and its
explicit region index table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# R-parity constants: R uses 1-based state indices (neutral=3 for i6,
# neutral=2 for i3). Python uses 0-based (neutral=2 for i6, neutral=1 for i3).
_NEUTRAL_IDX = {"i6": 2, "i3": 1}


def filter_high_p_normals(
    hmm_state_matrix: np.ndarray,
    bayesnet_result: dict,
    *,
    HMM_type: str = "i6",  # noqa: N803
    BayesMaxPNormal: float = 0.5,  # noqa: N803
    reassign: bool = False,
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
    reassign
        R ``reassignCNVs=TRUE`` (R default): after ``removeCNV``, every
        retained region is reassigned to the state with the highest
        posterior mean (``inferCNV_BayesNet.R:491-540``). Applied on the
        post-removal region list, matching R's ``obj@cnv_probabilities``
        subsetting order.

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

    if reassign:
        # R reassignCNV: for retained regions, argmax of the region-level
        # posterior mean vector (ties -> first max, same as R which()).
        highest = np.argmax(cnv_posterior, axis=1).astype(np.int8)
        keep = np.setdiff1d(np.arange(n_regions), to_remove, assume_unique=False)
        for r in keep:
            cell_idx = region_cell_idx[r]
            gene_idx = region_gene_idx[r]
            if cell_idx.size == 0 or gene_idx.size == 0:
                continue
            filtered[np.ix_(cell_idx, gene_idx)] = highest[r]

    return filtered


def summarize_bayes_regions(
    regions: pd.DataFrame,
    posterior: np.ndarray,
    *,
    HMM_type: str = "i6",
    BayesMaxPNormal: float = 0.5,
    reassign: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return retained regions and an unfiltered posterior index table.

    ``state_step17`` / ``state`` are zero-based; ``state_r`` is one-based.
    ``posterior_row`` indexes the original, unfiltered posterior array.
    Region IDs are scoped to one run and survive Bayes removal/reassignment.
    Bin endpoints are inclusive. i6 ``cn`` keeps the calibrated model levels
    (complete loss is modelled as 0.01); i3 keeps its signed delta convention.
    """
    if HMM_type not in _NEUTRAL_IDX:
        raise ValueError(f"Unknown HMM_type: {HMM_type!r}")
    levels = np.array([0.01, 0.5, 1.0, 1.5, 2.0, 3.0]) if HMM_type == "i6" else np.array([-1., 0., 1.])
    post = np.asarray(posterior, dtype=np.float64)
    if post.shape != (len(regions), len(levels)):
        raise ValueError("Region table and unfiltered posterior must have matching rows/states")
    if not np.isfinite(post).all() or np.any((post < 0) | (post > 1)):
        raise ValueError("Posterior probabilities must be finite and in [0, 1]")
    if len(post) and not np.allclose(post.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Posterior rows must sum to one")

    out = regions.copy().reset_index(drop=True)
    raw = out["state"].to_numpy(dtype=np.int64)
    if np.any(out["state"].to_numpy() != raw) or np.any((raw < 0) | (raw >= len(levels))):
        raise ValueError("Input region states must use zero-based HMM labels")
    out["posterior_row"] = np.arange(len(out), dtype=np.int64)
    out["region_id"] = [f"region_{i + 1:06d}" for i in range(len(out))]
    out["state_step17"] = raw
    out["p_normal"] = post[:, _NEUTRAL_IDX[HMM_type]]
    out["removed_by_pnormal"] = out["p_normal"] > BayesMaxPNormal
    states = post.argmax(axis=1) if reassign and len(post) else raw.copy()
    states[out["removed_by_pnormal"].to_numpy()] = _NEUTRAL_IDX[HMM_type]
    out["state"] = states
    out["state_r"] = states + 1
    out["cn"] = levels[states]
    for k in range(len(levels)):
        out[f"p_state_{k}"] = post[:, k]
    retained = out.loc[~out["removed_by_pnormal"]].copy().reset_index(drop=True)
    return retained, out


__all__ = ["filter_high_p_normals", "summarize_bayes_regions"]

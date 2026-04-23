"""InferCNVResult — output container for infercnv().

Schema is frozen now (G1 patch P1 + P2) so Phase 2/3 can fill fields
without breaking the Phase 1 AnnData persistence round-trip. All Phase 2/3
fields default to None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


@dataclass
class InferCNVResult:
    """Container for all pyinfercnv phases (Phase 2/3 fields optional).

    Phase 1 fields
    --------------
    chr_pos
        Ordered mapping chromosome -> start column index in cnv_matrix.
    cnv_matrix
        log2(FC)-space smoothed/centered/denoised matrix,
        shape (n_cells, n_bins), float32 C-order.
    cnv_matrix_fc
        Linear FC space = 2 ** cnv_matrix, same shape, centered around 1.0.
    cell_meta
        Per-cell metadata aligned with cnv_matrix rows. Must contain
        'is_reference' column.
    gene_values
        Optional per-gene CNV (when calculate_gene_values=True);
        shape (n_cells, n_genes).

    G1 patch P2 — raw reference handle
    ----------------------------------
    ref_counts_raw
        Raw integer counts (dense float32) for reference cells, captured
        BEFORE CPM normalize. Needed by Phase 2 calibrate_i6_emission.
        Shape (n_ref_cells, n_genes); dtype float32.

    Companion float64 field (transient)
    -----------------------------------
    cnv_matrix_f64
        Full-precision Phase 1 output, shape (n_cells, n_bins), float64.
        Populated by :func:`pyinfercnv.pipeline.infercnv` immediately
        before ``cnv_matrix`` is cast to float32 for the public contract.
        Used only by :mod:`pyinfercnv.pipeline_phase2` to avoid the
        precision drop at the Phase 1 → Phase 2 handoff; the caller (top-
        level ``infercnv()``) nulls this out after Phase 2 consumes it,
        so external consumers should not rely on it being present.
        Never persisted by :meth:`write_to_anndata`.

    Phase 2+ fields (None until those phases run)
    ---------------------------------------------
    subclusters
        Per-cell integer subcluster label, shape (n_cells,). int32.
    hmm_states
        6-state Viterbi output, shape (n_cells, n_bins). int8.
    hmm_states_i3
        3-state Viterbi output, shape (n_cells, n_bins). int8.
    cnv_regions
        Per-region BED-like DataFrame with columns
        [cell_group, chromosome, start, end, state, posterior_p_normal].
    posterior_p_normal
        Bayes filter output per (cell_group, region). DataFrame.
    profile
        Per-block wall-clock and RSS (psutil) captured during pipeline run
        when `debug=True`. Free-form dict.
    """

    chr_pos: dict[str, int]
    cnv_matrix: np.ndarray
    cnv_matrix_fc: np.ndarray
    cell_meta: pd.DataFrame
    gene_values: np.ndarray | None = None

    # G1 P2
    ref_counts_raw: np.ndarray | None = None

    # Phase 1 → Phase 2 float64 handoff (transient; nulled by pipeline after Phase 2)
    cnv_matrix_f64: np.ndarray | None = None

    # G1 P1 — Phase 2+ schema frozen
    subclusters: np.ndarray | None = None
    hmm_states: np.ndarray | None = None
    hmm_states_i3: np.ndarray | None = None
    cnv_regions: pd.DataFrame | None = None
    posterior_p_normal: pd.DataFrame | None = None

    profile: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.cnv_matrix.shape != self.cnv_matrix_fc.shape:
            raise ValueError(
                f"cnv_matrix shape {self.cnv_matrix.shape} != "
                f"cnv_matrix_fc shape {self.cnv_matrix_fc.shape}"
            )
        if len(self.cell_meta) != self.cnv_matrix.shape[0]:
            raise ValueError(
                f"cell_meta rows {len(self.cell_meta)} != "
                f"cnv_matrix rows {self.cnv_matrix.shape[0]}"
            )
        if "is_reference" not in self.cell_meta.columns:
            raise ValueError("cell_meta must contain 'is_reference' column")

    @property
    def n_cells(self) -> int:
        return int(self.cnv_matrix.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.cnv_matrix.shape[1])

    @property
    def chromosomes(self) -> list[str]:
        return list(self.chr_pos.keys())

    def write_to_anndata(self, adata: "AnnData", *, key_added: str = "cnv") -> None:
        """Persist all present fields into adata, aligning with infercnvpy conventions.

        Persistence layout (G1 P1 schema frozen now):

        - `adata.obsm[f"X_{key}"]` <- cnv_matrix (Phase 1)
        - `adata.uns[key]` dict with:
            - `chr_pos` (Phase 1)
            - `cnv_regions` (Phase 2/3, DataFrame)
            - `posterior_p_normal` (Phase 3, DataFrame)
            - `profile` (any phase, dict)
        - `adata.obs[f"{key}_subcluster"]` <- subclusters (Phase 2)
        - `adata.obsm[f"X_{key}_hmm_states"]` <- hmm_states (Phase 2 i6)
        - `adata.obsm[f"X_{key}_hmm_states_i3"]` <- hmm_states_i3 (Phase 2 i3)
        - `adata.layers[f"{key}_gene_values"]` <- gene_values (Phase 1 opt)
        - `adata.uns[f"{key}_ref_counts_raw"]` <- ref_counts_raw (G1 P2, Phase 1)
        """
        if adata.n_obs != self.n_cells:
            raise ValueError(
                f"adata.n_obs {adata.n_obs} != InferCNVResult.n_cells {self.n_cells}"
            )

        adata.obsm[f"X_{key_added}"] = self.cnv_matrix

        uns_entry: dict[str, Any] = {"chr_pos": self.chr_pos}
        if self.cnv_regions is not None:
            uns_entry["cnv_regions"] = self.cnv_regions
        if self.posterior_p_normal is not None:
            uns_entry["posterior_p_normal"] = self.posterior_p_normal
        if self.profile is not None:
            uns_entry["profile"] = self.profile
        adata.uns[key_added] = uns_entry

        if self.gene_values is not None:
            adata.layers[f"{key_added}_gene_values"] = self.gene_values
        if self.ref_counts_raw is not None:
            adata.uns[f"{key_added}_ref_counts_raw"] = self.ref_counts_raw
        if self.subclusters is not None:
            adata.obs[f"{key_added}_subcluster"] = self.subclusters
        if self.hmm_states is not None:
            adata.obsm[f"X_{key_added}_hmm_states"] = self.hmm_states
        if self.hmm_states_i3 is not None:
            adata.obsm[f"X_{key_added}_hmm_states_i3"] = self.hmm_states_i3

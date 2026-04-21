"""Hidden spike-in (hspike) calibration of i6 HMM emission parameters.

R-parity source map
-------------------
This module mirrors four R files in ``infercnv/infercnv-master/R``:

    inferCNV_hidden_spike.R
        ``.build_and_add_hspike`` (lines 3-165)     -> :func:`_build_hspike_counts`
        ``.get_hspike_chr_info`` (lines 170-215)    -> :data:`HSPIKE_CHR_INFO`
    inferCNV_meanVarSim.R
        ``.get_simulated_cell_matrix_using_meanvar_trend`` (lines 1-19)
                                                    -> :func:`_simulate_meanvar_counts`
        ``.get_mean_var_given_matrix`` (lines 189-211) -> :func:`_fit_meanvar_spline` (gene-level)
        ``.get_mean_vs_p0_table`` + ``.apply_dropout`` -> :func:`_apply_dropout`
    inferCNV_HMM.R
        ``get_spike_dists`` (lines 15-31) + ``.get_gene_expr_by_cnv`` (lines 45-68)
                                                    -> :func:`_gene_expr_mean_sd_by_cnv`
        ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit`` (lines 154-212)
                                                    -> :func:`_fit_cnv_sd_vs_num_cells_trend`

Algorithm overview
------------------
R's i6 HMM needs, per CNV state ``s in {0.01, 0.5, 1, 1.5, 2, 3}``:
    * a mean ``mu_s`` in log2-FC space
    * a per-``num_cells`` sd function ``sigma_s(n)``

so that :class:`pyinfercnv.hmm.i6.predict_i6` can be called on rowMeans
across a tumour subcluster with sd scaled by ``num_cells`` (R
``.get_state_emission_params`` in ``inferCNV_HMM.R:586-614``).

R obtains ``mu_s`` and ``sigma_s(n)`` by:

    1.  Building a synthetic infercnv object (``.hspike``) over a *fake*
        genome with 11 chromosomes (``HSPIKE_CHR_INFO``), alternating
        neutral blocks (cnv=1) with blocks at each target CN level; each
        block carries ``num_genes_per_chr`` fake genes.
    2.  For every reference-group gene-means profile ``gene_means`` (taken
        from the real normalized reference counts), simulating 100 normal
        cells + 100 spiked-tumour cells via ``sim_method in {meanvar,
        simple, splatter}``. Only ``'meanvar'`` is ported for Phase 2; the
        other two raise :class:`NotImplementedError` (``sim_method='simple'``
        and ``'splatter'`` require 500+ lines of additional R; deferred).
    3.  Running the full Phase 1 inferCNV pipeline (filter → normalize →
        log2 → subtract_ref → clip → smooth → center → subtract_ref₂ →
        invert? **no — stays in log2-FC**) on the hspike counts. The
        result ``hspike_log2fc`` has one row per simulated cell and one
        column per fake gene, so each hspike chromosome's contiguous gene
        columns carry the known CNV label of that block.
    4.  Computing ``mu_s = mean(hspike_log2fc[chr == s_name])`` and
        ``sigma_s = sd(...)`` across the fake-chr gene block (per-cell
        sample-level trend fitted separately).
    5.  Fitting ``lm(log(sigma_s(n)) ~ log(n))`` for ``n in 1..100``:
        for each n, resample n cells (with replacement from the block's
        expression values), compute their mean, repeat 100 rounds, take
        the sd over rounds — that's ``sigma_s(n)``.

The output of :func:`calibrate_i6_emission` is a frozen
:class:`HspikeCalibration` encapsulating ``mus``, ``sigmas`` (for n=1),
``sd_log_slope`` and ``sd_log_intercept`` (the lm fit), plus the fake-chr
CNV grid for provenance.

Matrix layout
-------------
* ``ref_counts_raw`` inputs are ``(n_ref_cells, n_genes_post_filter)`` dense
  float32, in post-Phase-1-filter gene order (from
  :attr:`InferCNVResult.ref_counts_raw`).
* ``ref_groups_local`` keys into rows of ``ref_counts_raw`` (0-indexed).
  Callers must remap from global cell indices before invoking this module.
* Simulated hspike matrix is ``(n_hspike_cells, n_hspike_genes)`` float32,
  not sparse — hspike is always small (~2 * n_groups * num_cells_per_state
  cells × 11 * num_genes_per_chr genes ≈ 200 × 4400 for defaults).
* Internal Phase 1 replay on hspike produces a dense float32
  ``(n_hspike_cells, n_hspike_genes)`` in log2-FC space.

Tier classification
-------------------
Per ``docs/superpowers/specs/2026-04-21-pyinfercnv-master-design.md`` §5.2,
hspike calibration is **tier-3.5 empirical** — the NB-with-dropout
simulation + smooth.spline mean-var fit depart from R on numerical trivia
(RNG streams, spline knots), so bit-exact parity is not targeted. Acceptance
metric: each ``mu_s`` within 10% of R's value on the shared fixture, and
subcluster Viterbi labelling ARI ≥ 0.85 against R on the empirical floor.

G1 patches in scope
-------------------
* **P3** — :func:`calibrate_i6_emission` must be registered in
  ``pyinfercnv.hmm.__init__.__all__``.
* **P11** — optional ``profile`` dict parameter receives psutil timings per
  block (simulate / Phase1-replay / mean-sd / trend-lm).

What this module does NOT do
----------------------------
* No call to :func:`pyinfercnv.pipeline.infercnv`. Replay uses the lower-level
  step functions directly (``filter_low_expression_genes``, ``normalize_by_seq_depth``,
  ``log2_plus1``, ``subtract_reference``, ``apply_max_centered_threshold``,
  ``smooth_pyramidinal``, ``center_cells``) to avoid re-extracting from an
  AnnData and to keep the fake-chr layout under our direct control.
* No handling of ``sim_method in {'simple', 'splatter'}``. NotImplementedError.
* No persistence into AnnData. The caller (Phase 2 pipeline) threads the
  returned :class:`HspikeCalibration` into :func:`pyinfercnv.hmm.i6.predict_i6`.

No new numba kernels
--------------------
All compiled hot loops in Phase 2 stay inside
:mod:`pyinfercnv.kernels.hmm_viterbi_numba`. The hspike body relies on
NumPy vectorization + SciPy primitives (``scipy.interpolate``,
``scipy.special``) only; no ``@njit`` is added here or in
:mod:`pyinfercnv.pipeline_phase2`. (G2 Q4 documentation.)

Skeleton status
---------------
This file is a G2-pending API skeleton. Every function body raises
``NotImplementedError("skeleton — G2 pending")``. Implementation lands in
Wave 2 after codex G2 adjudication on API + data layout + psutil hook
placement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from pyinfercnv.config import InferCNVConfig


# --------------------------------------------------------------------------- #
# Constants (mirror R)                                                        #
# --------------------------------------------------------------------------- #

#: R ``.get_hspike_chr_info`` CNV grid: 11 fake chromosomes alternating
#: neutral (cnv=1) and at-target (cnv in the i6 level set). Listed in the
#: canonical R order so the hspike column layout is deterministic.
HSPIKE_CHR_INFO: tuple[tuple[str, float], ...] = (
    ("chrA",     1.0),
    ("chr_0",    0.01),
    ("chr_B",    1.0),
    ("chr_0pt5", 0.5),
    ("chr_C",    1.0),
    ("chr_1pt5", 1.5),
    ("chr_D",    1.0),
    ("chr_2pt0", 2.0),
    ("chr_E",    1.0),
    ("chr_3pt0", 3.0),
    ("chr_F",    1.0),
)

#: i6 CNV levels for which we report calibrated emission parameters
#: (indexed identically to :data:`pyinfercnv.hmm.i6.I6_CNV_LEVELS`).
I6_CNV_LEVELS_CALIBRATED: NDArray[np.float64] = np.array(
    [0.01, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64
)

#: R ``.build_and_add_hspike`` defaults.
HSPIKE_DEFAULT_NUM_CELLS_PER_STATE: int = 100
HSPIKE_DEFAULT_NUM_GENES_PER_CHR: int = 400

#: R ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit`` defaults.
HSPIKE_TREND_NUM_ROUNDS: int = 100
HSPIKE_TREND_MAX_NUM_CELLS: int = 100


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HspikeCalibration:
    """Emission calibration for i6 HMM, derived from the hidden spike-in.

    Mirrors the R pair ``(cnv_mean_sd, cnv_level_to_mean_sd_fit)`` that
    :func:`pyinfercnv.pipeline_phase2.run_phase2` passes to the HMM layer.
    Frozen so it's safe to share across threads and cache between calls.

    Attributes
    ----------
    cnv_levels
        Shape (6,) — the raw CN levels corresponding to each state index:
        ``[0.01, 0.5, 1.0, 1.5, 2.0, 3.0]``. Included explicitly so the
        caller can label outputs without hard-coding the grid.
    state_mus
        Shape (6,) float64 — per-state mean in log2-FC space, as read from
        the post-Phase-1 hspike matrix restricted to that state's fake-chr
        block. Aligns with :data:`pyinfercnv.hmm.i6.I6_DEFAULT_MUS` in shape
        and semantics but with R-calibrated values.
    state_sigmas
        Shape (6,) float64 — per-state sd at ``num_cells=1`` (i.e., the raw
        per-gene-value spread within the fake-chr block). For subcluster
        HMM calls, use :meth:`sigmas_for_num_cells` to apply the trend fit.
    sd_log_slope, sd_log_intercept
        Shape (6,) float64 each — per-state lm coefficients for
        ``log(sigma(n)) ~ a * log(n) + b``. Slope is typically near ``-0.5``
        (sd of a sample mean scales as ``1/sqrt(n)``); intercept equals
        ``log(state_sigmas)`` in the limit.
    hspike_log2fc
        Optional, shape (n_hspike_cells, n_hspike_genes) float32 — the full
        post-Phase-1 hspike matrix retained for diagnostics/G3 verification.
        ``None`` when :func:`calibrate_i6_emission` is invoked with
        ``keep_matrix=False`` (default True during development).
    gene_chr_labels
        Optional, shape (n_hspike_genes,) of object dtype — fake-chr name per
        gene column of ``hspike_log2fc``. Same ``None`` rule as above.

    Methods
    -------
    sigmas_for_num_cells(n)
        Returns shape (6,) sd vector adjusted for a subcluster of ``n``
        cells, computed as ``exp(sd_log_slope * log(n) + sd_log_intercept)``.
        Used by Phase 2 pipeline when calling
        :func:`pyinfercnv.hmm.i6.predict_i6` on a subcluster's rowMeans.
    """

    cnv_levels: NDArray[np.float64]
    state_mus: NDArray[np.float64]
    state_sigmas: NDArray[np.float64]
    sd_log_slope: NDArray[np.float64]
    sd_log_intercept: NDArray[np.float64]
    hspike_log2fc: NDArray[np.float32] | None = None
    gene_chr_labels: NDArray[np.object_] | None = None

    def sigmas_for_num_cells(self, num_cells: int) -> NDArray[np.float64]:
        """Return shape (6,) sd predicted for a subcluster of ``num_cells``.

        Mirror of R ``.get_state_emission_params`` (``inferCNV_HMM.R:586``).
        Uses the stored ``(sd_log_slope, sd_log_intercept)`` lm fit. Never
        returns negative or zero; callers should trust the result.
        """
        raise NotImplementedError("skeleton — G2 pending")


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def calibrate_i6_emission(
    ref_counts_raw: NDArray[np.float32],
    ref_groups_local: Mapping[str, NDArray[np.intp]] | None,
    *,
    config: "InferCNVConfig | None" = None,
    sim_method: Literal["meanvar", "simple", "splatter"] = "meanvar",
    aggregate_normals: bool = False,
    num_cells_per_state: int = HSPIKE_DEFAULT_NUM_CELLS_PER_STATE,
    num_genes_per_chr: int = HSPIKE_DEFAULT_NUM_GENES_PER_CHR,
    trend_num_rounds: int = HSPIKE_TREND_NUM_ROUNDS,
    trend_max_num_cells: int = HSPIKE_TREND_MAX_NUM_CELLS,
    include_dropout: bool = True,
    keep_matrix: bool = False,
    random_state: int | None = 0,
    profile: dict[str, Any] | None = None,
) -> HspikeCalibration:
    """Top-level R-parity entry for i6 emission calibration.

    Executes the full ``.build_and_add_hspike`` → full-Phase-1-replay →
    ``get_spike_dists`` → ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit``
    pipeline and returns a frozen :class:`HspikeCalibration`.

    Parameters
    ----------
    ref_counts_raw
        Shape (n_ref_cells, n_genes_post_filter) dense float32 — raw (pre-CPM,
        pre-log) counts of reference cells after Phase 1 gene filtering.
        Directly consumed from :attr:`InferCNVResult.ref_counts_raw`.
    ref_groups_local
        Keys: reference group names (e.g. T-cell, B-cell). Values: integer
        arrays indexing **into rows of ref_counts_raw** (0-based, local to
        the reference matrix, NOT global adata.obs indices). ``None`` is
        equivalent to ``{'normalsToUse': arange(n_ref_cells)}`` (reference-
        less mode, mirrors R's ``observation_grouped_cell_indices`` fallback
        in ``inferCNV_hidden_spike.R:19-26``).
    config
        Phase 1 :class:`InferCNVConfig`. Used to re-run the Phase 1 pipeline
        on simulated hspike counts. ``None`` uses defaults. Only the
        preprocess/smooth/center/subtract/threshold fields are honoured;
        HMM fields are ignored.
    sim_method
        Only ``"meanvar"`` is implemented in Phase 2; other values raise
        :class:`NotImplementedError`.
    aggregate_normals
        If ``True``, collapse all reference groups into a single pseudo-group
        ``'normalsToUse'`` (mirrors R ``aggregate_normals=TRUE`` branch in
        ``inferCNV_hidden_spike.R:11-13``). Default ``False`` matches R's
        per-group simulation.
    num_cells_per_state
        Per R, 100 cells are simulated for each CN state *per reference
        group*. Tune down for unit tests.
    num_genes_per_chr
        Per R, 400 genes per fake chromosome. ``chr_F`` receives the
        remainder so the total hspike gene count equals
        ``n_genes_post_filter`` of the real matrix (R fakes a genome with
        ``n_genes_post_filter`` total genes).
    trend_num_rounds, trend_max_num_cells
        R defaults (100 rounds × n=1..100). Used by
        :func:`_fit_cnv_sd_vs_num_cells_trend`.
    include_dropout
        Whether to apply the logistic-dropout post-processing to the
        simulated counts. Python-safe alias of R's dotted kwarg
        ``include.dropout`` (R ``inferCNV_hidden_spike.R:90/93/126/129``);
        Python PEP-8 disallows dots in identifiers so the underscore
        spelling is canonical here (G2 Q2).
    keep_matrix
        If ``True``, retain ``hspike_log2fc`` + ``gene_chr_labels`` on the
        returned :class:`HspikeCalibration` for diagnostics. Default
        ``False`` per G2 Q5: production runs allocate only the calibration
        vectors; set ``True`` for debugging / G3 parity inspection.
    random_state
        Passed through to :class:`numpy.random.default_rng`. R uses an
        implicit ``set.seed(...)`` — we make it explicit and deterministic.
    profile
        Optional dict mutated in place with G1-P11 psutil timings under
        keys ``hspike_sim``, ``hspike_phase1_replay``, ``hspike_get_dists``,
        ``hspike_trend_lm``. ``None`` skips profiling.

    Returns
    -------
    HspikeCalibration
        Frozen calibration to pass to :func:`pyinfercnv.hmm.i6.predict_i6`
        directly (no subcluster aggregation) or via
        :meth:`HspikeCalibration.sigmas_for_num_cells` when the caller does
        per-subcluster rowMeans-then-HMM (R's
        ``predict_CNV_via_HMM_on_tumor_subclusters`` path).

    Raises
    ------
    NotImplementedError
        When ``sim_method in {'simple', 'splatter'}`` — those variants
        require ~500 lines of additional R to port (``inferCNV_simple_sim.R``
        + ``SplatterScrape.R``) and are deferred past Phase 2.
    ValueError
        When ``ref_counts_raw`` is empty, ``ref_groups_local`` values go
        out of bounds, or ``num_cells_per_state < 2`` (need ≥2 cells to
        compute per-group ``gene_means``).
    """
    raise NotImplementedError("skeleton — G2 pending")


# --------------------------------------------------------------------------- #
# Private helpers (stable signatures for G2 review; bodies in Wave 2)         #
# --------------------------------------------------------------------------- #


def _build_hspike_counts(
    ref_counts_raw: NDArray[np.float32],
    ref_groups_local: Mapping[str, NDArray[np.intp]],
    *,
    num_cells_per_state: int,
    num_genes_per_chr: int,
    sim_method: Literal["meanvar", "simple", "splatter"],
    include_dropout: bool,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.object_], dict[str, NDArray[np.intp]], dict[str, NDArray[np.intp]]]:
    """Build the synthetic hspike counts matrix mirroring R ``.build_and_add_hspike``.

    Returns
    -------
    counts
        Shape (n_hspike_cells, n_hspike_genes) float32. Cells ordered as:
        for each reference group, first ``num_cells_per_state`` simulated
        normals (spike_norm), then ``num_cells_per_state`` simulated
        spiked-tumour cells (spike_tumor). Then repeat per group. Matches
        R ``cbind`` order at ``inferCNV_hidden_spike.R:138-142``.
    gene_chr_labels
        Shape (n_hspike_genes,) object dtype — per-gene fake chromosome
        name drawn from :data:`HSPIKE_CHR_INFO`. Used later by
        :func:`_gene_expr_mean_sd_by_cnv` to partition columns.
    reference_indices, observation_indices
        Per-group row-index arrays into ``counts``. Mirror R
        ``reference_grouped_cell_indices`` and
        ``observation_grouped_cell_indices`` (R 1-based, Python 0-based).
        Needed by the Phase 1 replay step so ``subtract_reference`` can use
        the spike-norm cells as the reference pool.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _fit_meanvar_spline(
    counts: NDArray[np.float32],
    cell_groupings: Mapping[str, NDArray[np.intp]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit R ``smooth.spline(log(v+1) ~ log(m+1))`` gene-wise mean-variance trend.

    Returns ``(knots_log_m, knots_log_v)`` — the spline evaluated on a dense
    grid; :func:`_simulate_meanvar_counts` uses ``np.interp`` against it.
    ``scipy.interpolate.UnivariateSpline`` with R-matching smoothing
    parameter is the implementation target; the tuple signature avoids
    leaking a live spline object across calls.

    R reference: ``inferCNV_meanVarSim.R:30`` (``smooth.spline``) +
    ``inferCNV_meanVarSim.R:189-211`` (``.get_mean_var_given_matrix``).
    """
    raise NotImplementedError("skeleton — G2 pending")


def _simulate_meanvar_counts(
    gene_means: NDArray[np.float64],
    spline_knots: tuple[NDArray[np.float64], NDArray[np.float64]],
    num_cells: int,
    dropout_logistic_params: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Sample ``(num_cells, n_genes)`` counts from the mean-var trend.

    Per-gene: ``var = max(exp(spline(log(m+1))) - 1, 0)``, then draw
    ``round(max(N(m, sqrt(var)), 0))`` (R
    ``inferCNV_meanVarSim.R:105-119``). Optionally apply logistic dropout
    via ``_apply_dropout`` if ``dropout_logistic_params`` is not None.

    Returns
    -------
    simulated
        Shape ``(num_cells, len(gene_means))`` float32.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _apply_dropout(
    counts: NDArray[np.float32],
    dropout_logistic_params: tuple[NDArray[np.float64], NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Logistic-dropout post-processing, mirror of R ``.apply_dropout``.

    R reference: ``inferCNV_meanVarSim.R:122-161``. The ``padj`` adjustment
    line 137 must be preserved verbatim for R-parity.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _run_phase1_replay_on_hspike(
    hspike_counts: NDArray[np.float32],
    *,
    gene_chr_labels: NDArray[np.object_],
    reference_indices: dict[str, NDArray[np.intp]],
    config: "InferCNVConfig",
) -> NDArray[np.float32]:
    """Re-run the Phase 1 log2-FC pipeline on the hspike counts.

    Steps (mirror Phase 1 pipeline.py ``infercnv()``):
        * filter_low_expression_genes on fake genome
          (cutoff reused from ``config.cutoff`` for audit parity)
        * normalize_by_seq_depth (CPM)
        * log2_plus1
        * subtract_reference (use hspike's ``reference_indices`` as refs)
        * apply_max_centered_threshold
        * smooth_pyramidinal per fake chromosome
          (blocks defined by ``gene_chr_labels``)
        * center_cells (median)
        * subtract_reference (2nd pass)

    Does NOT invert_log2 — the calibration lives in log2-FC space to match
    R's ``get_spike_dists`` (which reads ``hspike@expr.data`` directly at
    ``inferCNV_HMM.R:51``, pre-invert).

    Returns
    -------
    hspike_log2fc
        Shape ``(n_hspike_cells, n_hspike_genes_post_filter)`` float32.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _gene_expr_mean_sd_by_cnv(
    hspike_log2fc: NDArray[np.float32],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract per-CNV-state ``(mu, sigma)`` from the Phase-1-replayed hspike.

    Mirror of R ``get_spike_dists`` + ``.get_gene_expr_by_cnv``
    (``inferCNV_HMM.R:15-68``). Pools values across observation cells (the
    spike-tumour cells) and across all fake chromosomes sharing the same
    CN level. Returns aligned (6,) arrays for :data:`I6_CNV_LEVELS_CALIBRATED`.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _fit_cnv_sd_vs_num_cells_trend(
    hspike_log2fc: NDArray[np.float32],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
    *,
    num_rounds: int,
    max_num_cells: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit ``lm(log(sigma(n)) ~ log(n))`` per CNV state.

    Mirror of R ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit``
    (``inferCNV_HMM.R:154-212``). For each CN state and each ``n in 1..max_num_cells``,
    draw ``num_rounds`` resamples of size ``n`` with replacement from the
    pooled expression values of that CN block, take each resample's mean,
    then compute sd across rounds — that's ``sigma_s(n)``. Fit linear
    regression ``log(sigma_s(n)) = a * log(n) + b`` via closed-form OLS.

    Returns
    -------
    slope, intercept
        Shape (6,) float64 each — per-state lm coefficients. Slope should
        be close to -0.5 (central-limit scaling) on well-calibrated hspikes.
    """
    raise NotImplementedError("skeleton — G2 pending")


__all__ = [
    "HspikeCalibration",
    "calibrate_i6_emission",
    "HSPIKE_CHR_INFO",
    "I6_CNV_LEVELS_CALIBRATED",
    "HSPIKE_DEFAULT_NUM_CELLS_PER_STATE",
    "HSPIKE_DEFAULT_NUM_GENES_PER_CHR",
    "HSPIKE_TREND_NUM_ROUNDS",
    "HSPIKE_TREND_MAX_NUM_CELLS",
]

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

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline
from scipy.special import expit

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
        n = max(int(num_cells), 1)
        result = np.exp(self.sd_log_slope * np.log(n) + self.sd_log_intercept)
        return np.maximum(result, 1e-6).astype(np.float64)


# --------------------------------------------------------------------------- #
# Profile helper (mirrors pipeline.py pattern)                               #
# --------------------------------------------------------------------------- #


def _rss_mb() -> float | None:
    """Return current process RSS in MB via psutil, or None if unavailable."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _record_profile(
    profile: dict[str, Any] | None,
    key: str,
    t0: float,
    rss_before: float | None,
) -> None:
    if profile is None:
        return
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    profile[key] = {
        "wallclock_s": float(elapsed),
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_delta_mb": (
            (rss_after - rss_before)
            if (rss_after is not None and rss_before is not None)
            else None
        ),
    }


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
    # --- edge-case validation (G2 Q7) ---
    if sim_method in ("simple", "splatter"):
        raise NotImplementedError(
            f"sim_method={sim_method!r} is not implemented in Phase 2. "
            "Only 'meanvar' is ported. 'simple' and 'splatter' require "
            "inferCNV_simple_sim.R / SplatterScrape.R (~500 lines) and are deferred."
        )

    ref_counts_raw = np.asarray(ref_counts_raw, dtype=np.float32)
    if ref_counts_raw.size == 0 or ref_counts_raw.shape[0] == 0:
        raise ValueError(
            "ref_counts_raw is empty (0 cells). Cannot calibrate hspike emission."
        )

    n_ref_cells = ref_counts_raw.shape[0]

    if num_cells_per_state < 2:
        raise ValueError(
            f"num_cells_per_state must be >= 2, got {num_cells_per_state}."
        )

    # Build / normalise ref groups
    if ref_groups_local is None:
        ref_groups: Mapping[str, NDArray[np.intp]] = {
            "normalsToUse": np.arange(n_ref_cells, dtype=np.intp)
        }
    else:
        ref_groups = ref_groups_local

    if aggregate_normals:
        all_idx = np.concatenate([np.asarray(v, dtype=np.intp) for v in ref_groups.values()])
        ref_groups = {"normalsToUse": all_idx}

    # Validate each group has >=2 cells (G2 Q7)
    for grp_name, idx_arr in ref_groups.items():
        arr = np.asarray(idx_arr, dtype=np.intp)
        if arr.shape[0] < 2:
            raise ValueError(
                f"Reference group {grp_name!r} has {arr.shape[0]} cell(s); "
                "need >= 2 to compute gene means."
            )

    from pyinfercnv.config import InferCNVConfig as _Config
    cfg = config if config is not None else _Config()

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    # Stage 1: build hspike counts                                        #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    hspike_counts, gene_chr_labels_full, reference_indices, observation_indices = (
        _build_hspike_counts(
            ref_counts_raw,
            ref_groups,
            num_cells_per_state=num_cells_per_state,
            num_genes_per_chr=num_genes_per_chr,
            sim_method=sim_method,
            include_dropout=include_dropout,
            rng=rng,
        )
    )

    _record_profile(profile, "hspike_sim", t0, rss0)

    # ------------------------------------------------------------------ #
    # Stage 2: Phase 1 replay                                             #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    hspike_log2fc, gene_chr_labels_filtered = _run_phase1_replay_on_hspike(
        hspike_counts,
        gene_chr_labels=gene_chr_labels_full,
        reference_indices=reference_indices,
        config=cfg,
    )

    _record_profile(profile, "hspike_phase1_replay", t0, rss0)

    # ------------------------------------------------------------------ #
    # Stage 3: get per-CNV mu/sigma                                       #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    state_mus, state_sigmas = _gene_expr_mean_sd_by_cnv(
        hspike_log2fc,
        gene_chr_labels_filtered,
        observation_indices,
    )

    _record_profile(profile, "hspike_get_dists", t0, rss0)

    # ------------------------------------------------------------------ #
    # Stage 4: trend LM fit                                               #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    sd_log_slope, sd_log_intercept = _fit_cnv_sd_vs_num_cells_trend(
        hspike_log2fc,
        gene_chr_labels_filtered,
        observation_indices,
        num_rounds=trend_num_rounds,
        max_num_cells=trend_max_num_cells,
        rng=rng,
    )

    _record_profile(profile, "hspike_trend_lm", t0, rss0)

    # ------------------------------------------------------------------ #
    # Assemble result                                                     #
    # ------------------------------------------------------------------ #
    return HspikeCalibration(
        cnv_levels=I6_CNV_LEVELS_CALIBRATED.copy(),
        state_mus=state_mus,
        state_sigmas=state_sigmas,
        sd_log_slope=sd_log_slope,
        sd_log_intercept=sd_log_intercept,
        hspike_log2fc=hspike_log2fc if keep_matrix else None,
        gene_chr_labels=(
            gene_chr_labels_filtered if keep_matrix else None
        ),
    )


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
    if sim_method in ("simple", "splatter"):
        raise NotImplementedError(
            f"sim_method={sim_method!r} is not implemented. Only 'meanvar' is supported."
        )

    n_real_genes = ref_counts_raw.shape[1]

    # Build fake chromosome layout: compute n_genes per chr (R .get_hspike_chr_info)
    n_remaining = n_real_genes - 10 * num_genes_per_chr
    if n_remaining < num_genes_per_chr:
        n_remaining = num_genes_per_chr

    chr_ngenes: list[int] = []
    chr_names: list[str] = []
    chr_cnvs: list[float] = []
    for i, (cname, ccnv) in enumerate(HSPIKE_CHR_INFO):
        if cname == "chr_F":
            n_g = n_remaining
        else:
            n_g = num_genes_per_chr
        chr_ngenes.append(n_g)
        chr_names.append(cname)
        chr_cnvs.append(ccnv)

    n_fake_genes = sum(chr_ngenes)

    # gene_chr_labels: per-gene fake chr name
    gene_chr_labels = np.empty(n_fake_genes, dtype=object)
    offset = 0
    for cname, n_g in zip(chr_names, chr_ngenes):
        gene_chr_labels[offset: offset + n_g] = cname
        offset += n_g

    # Sample gene indices from real data (R: sample(seq_len(nrow), size=num_genes, replace=TRUE))
    # Done once, shared across all reference groups
    genes_means_use_idx = rng.choice(n_real_genes, size=n_fake_genes, replace=True)

    # Fit mean-var spline from reference data (used for all groups)
    spline_knots = _fit_meanvar_spline(ref_counts_raw, None)

    # Fit dropout logistic params from reference data if needed
    dropout_logistic_params: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if include_dropout:
        dropout_logistic_params = _fit_dropout_logistic_params(ref_counts_raw)

    # Build cells matrix: per group, normals first then spiked tumor
    cells_list: list[NDArray[np.float32]] = []
    reference_indices: dict[str, NDArray[np.intp]] = {}
    observation_indices: dict[str, NDArray[np.intp]] = {}
    cell_counter = 0

    for normal_type, cell_idx in ref_groups_local.items():
        cell_idx_arr = np.asarray(cell_idx, dtype=np.intp)
        normal_cells_expr = ref_counts_raw[cell_idx_arr, :]  # (n_cells, n_real_genes)

        # per-gene means from this group
        gene_means_orig = normal_cells_expr.mean(axis=0).astype(np.float64)
        gene_means = gene_means_orig[genes_means_use_idx].copy()
        gene_means[gene_means == 0] = 1e-3  # avoid zeros

        # Simulate normal cells (no CNV scaling)
        sim_normal = _simulate_meanvar_counts(
            gene_means,
            spline_knots,
            num_cells_per_state,
            dropout_logistic_params,
            rng,
        )

        # Compute spiked gene means (multiply by CNV)
        hspike_gene_means = gene_means.copy()
        offset = 0
        for cname, n_g, ccnv in zip(chr_names, chr_ngenes, chr_cnvs):
            if ccnv != 1.0:
                hspike_gene_means[offset: offset + n_g] *= ccnv
            offset += n_g

        # Simulate spiked tumor cells
        sim_tumor = _simulate_meanvar_counts(
            hspike_gene_means,
            spline_knots,
            num_cells_per_state,
            dropout_logistic_params,
            rng,
        )

        # Record indices (0-based Python, matching the cbind order)
        spike_norm_name = f"simnorm_cell_{normal_type}"
        spike_tumor_name = f"spike_tumor_cell_{normal_type}"

        reference_indices[spike_norm_name] = np.arange(
            cell_counter, cell_counter + num_cells_per_state, dtype=np.intp
        )
        cell_counter += num_cells_per_state

        observation_indices[spike_tumor_name] = np.arange(
            cell_counter, cell_counter + num_cells_per_state, dtype=np.intp
        )
        cell_counter += num_cells_per_state

        cells_list.append(sim_normal)
        cells_list.append(sim_tumor)

    counts = np.concatenate(cells_list, axis=0)  # (n_hspike_cells, n_fake_genes)
    return counts, gene_chr_labels, reference_indices, observation_indices


def _fit_dropout_logistic_params(
    counts: NDArray[np.float32],
    cell_groupings: Mapping[str, NDArray[np.intp]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit dropout probability vs log(mean) from the reference matrix.

    Returns ``(log_m_grid, dropout_p_grid)`` for ``np.interp`` lookup.
    R reference: ``inferCNV_meanVarSim.R`` ``.get_mean_vs_p0_table`` +
    ``.get_logistic_params``.
    """
    counts_f = np.asarray(counts, dtype=np.float64)
    if cell_groupings is None:
        m = counts_f.mean(axis=0)
        p0 = (counts_f == 0).mean(axis=0)
    else:
        m_list: list[NDArray[np.float64]] = []
        p0_list: list[NDArray[np.float64]] = []
        for idx in cell_groupings.values():
            sub = counts_f[np.asarray(idx, dtype=np.intp), :]
            m_list.append(sub.mean(axis=0))
            p0_list.append((sub == 0).mean(axis=0))
        m = np.concatenate(m_list)
        p0 = np.concatenate(p0_list)

    # Fit spline: smooth p0 ~ log(m+1)
    log_m = np.log(m + 1.0)
    order = np.argsort(log_m)
    log_m_sorted = log_m[order]
    p0_sorted = np.clip(p0[order], 0.0, 1.0)

    # Remove duplicates for spline fitting
    _, unique_idx = np.unique(log_m_sorted, return_index=True)
    x_u = log_m_sorted[unique_idx]
    y_u = p0_sorted[unique_idx]

    if x_u.shape[0] < 4:
        # Not enough data: constant 0 dropout
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    try:
        spl = UnivariateSpline(x_u, y_u, k=min(3, len(x_u) - 1), s=len(x_u))
        log_m_grid = np.linspace(x_u[0], x_u[-1], max(100, len(x_u)))
        dropout_p_grid = np.clip(spl(log_m_grid), 0.0, 1.0)
    except Exception:
        log_m_grid = x_u
        dropout_p_grid = np.clip(y_u, 0.0, 1.0)

    return log_m_grid.astype(np.float64), dropout_p_grid.astype(np.float64)


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
    counts_f = np.asarray(counts, dtype=np.float64)
    n_cells, n_genes = counts_f.shape

    if cell_groupings is None:
        groupings: Mapping[str, NDArray[np.intp]] = {
            "allcells": np.arange(n_cells, dtype=np.intp)
        }
    else:
        groupings = cell_groupings

    m_all: list[NDArray[np.float64]] = []
    v_all: list[NDArray[np.float64]] = []

    for grp_idx in groupings.values():
        idx = np.asarray(grp_idx, dtype=np.intp)
        sub = counts_f[idx, :]
        m_grp = sub.mean(axis=0)
        # R var() uses ddof=1
        v_grp = sub.var(axis=0, ddof=1) if sub.shape[0] > 1 else np.zeros(n_genes)
        m_all.append(m_grp)
        v_all.append(v_grp)

    m = np.concatenate(m_all)
    v = np.concatenate(v_all)

    log_m = np.log(m + 1.0)
    log_v = np.log(v + 1.0)

    order = np.argsort(log_m)
    log_m_sorted = log_m[order]
    log_v_sorted = log_v[order]

    # Remove duplicates
    _, unique_idx = np.unique(log_m_sorted, return_index=True)
    x_u = log_m_sorted[unique_idx]
    y_u = log_v_sorted[unique_idx]

    if x_u.shape[0] < 4:
        # Degenerate: return trivial flat spline
        return x_u.astype(np.float64), y_u.astype(np.float64)

    # scipy UnivariateSpline mirrors R smooth.spline behaviour (G2 Q3)
    try:
        spl = UnivariateSpline(x_u, y_u, k=min(3, len(x_u) - 1), s=len(x_u))
        log_m_grid = np.linspace(x_u[0], x_u[-1], max(200, len(x_u)))
        log_v_grid = spl(log_m_grid)
    except Exception:
        log_m_grid = x_u
        log_v_grid = y_u

    return log_m_grid.astype(np.float64), log_v_grid.astype(np.float64)


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
    log_m_grid, log_v_grid = spline_knots
    gene_means_f = np.asarray(gene_means, dtype=np.float64)
    n_genes = gene_means_f.shape[0]

    # Vectorized: predict log_var for all genes at once
    log_m_genes = np.log(gene_means_f + 1.0)
    pred_log_v = np.interp(log_m_genes, log_m_grid, log_v_grid)
    gene_var = np.maximum(np.exp(pred_log_v) - 1.0, 0.0)  # (n_genes,)
    gene_sd = np.sqrt(gene_var)  # (n_genes,)

    # Draw normal random matrix in one shot
    mu = gene_means_f[None, :]   # (1, n_genes)
    sd = gene_sd[None, :]        # (1, n_genes)
    x = rng.normal(loc=mu, scale=sd, size=(num_cells, n_genes))
    x = np.maximum(x, 0.0)
    x = np.round(x).astype(np.float32)

    if dropout_logistic_params is not None:
        x = _apply_dropout(x, dropout_logistic_params, rng)

    return x


def _apply_dropout(
    counts: NDArray[np.float32],
    dropout_logistic_params: tuple[NDArray[np.float64], NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Logistic-dropout post-processing, mirror of R ``.apply_dropout``.

    R reference: ``inferCNV_meanVarSim.R:122-161``. The ``padj`` adjustment
    line 137 must be preserved verbatim for R-parity.
    """
    log_m_grid, dropout_p_grid = dropout_logistic_params
    counts_f = np.asarray(counts, dtype=np.float32)
    # R's apply(counts.matrix, 1, ...) is per-row of the R (genes x cells) matrix
    # = per column of our Python (cells x genes) matrix
    # So x = one gene's values across all cells

    ntotal = counts_f.shape[0]  # n_cells (per gene)

    # Per-gene statistics
    mean_per_gene = counts_f.mean(axis=0).astype(np.float64)  # (n_genes,)
    nzeros_per_gene = (counts_f == 0).sum(axis=0).astype(np.float64)  # (n_genes,)
    nremaining_per_gene = ntotal - nzeros_per_gene

    # Lookup dropout probability
    log_mean_per_gene = np.log(np.maximum(mean_per_gene, 1e-15))
    dropout_prob = np.interp(log_mean_per_gene, log_m_grid, dropout_p_grid)

    # R padj formula (line 137, preserved verbatim):
    # padj = ( (dropout_prob * ntotal) - nzeros ) / nremaining
    # padj = max(padj, 0)
    padj = (dropout_prob * ntotal - nzeros_per_gene) / np.maximum(nremaining_per_gene, 1.0)
    padj = np.maximum(padj, 0.0)  # (n_genes,)

    # Apply per-cell coin flip, only on non-zero entries
    u = rng.random(counts_f.shape)  # (n_cells, n_genes)
    drop_mask = (u < padj[None, :]) & (counts_f != 0)
    out = np.where(drop_mask, 0.0, counts_f).astype(np.float32)
    return out


def _run_phase1_replay_on_hspike(
    hspike_counts: NDArray[np.float32],
    *,
    gene_chr_labels: NDArray[np.object_],
    reference_indices: dict[str, NDArray[np.intp]],
    config: "InferCNVConfig",
) -> tuple[NDArray[np.float32], NDArray[np.object_]]:
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
    gene_chr_labels_filtered
        Shape ``(n_hspike_genes_post_filter,)`` object dtype — aligned with
        the filtered gene columns.
    """
    from pyinfercnv.preprocess import (  # noqa: PLC0415
        apply_max_centered_threshold,
        filter_low_expression_genes,
        log2_plus1,
        normalize_by_seq_depth,
        subtract_reference,
    )
    from pyinfercnv.smooth import smooth_pyramidinal  # noqa: PLC0415
    from pyinfercnv.center import center_cells  # noqa: PLC0415

    X = np.asarray(hspike_counts, dtype=np.float32)

    # --- Step 1: filter low-expression genes ---
    ref_idx_all = np.concatenate([
        np.asarray(v, dtype=np.intp) for v in reference_indices.values()
    ])
    keep_mask = filter_low_expression_genes(
        X,
        cutoff=config.cutoff,
        min_cells_per_gene=config.min_cells_per_gene,
        reference_cell_idx=ref_idx_all,
    )
    # If nothing passes the filter, keep all genes (avoid empty matrix)
    if not keep_mask.any():
        keep_mask = np.ones(X.shape[1], dtype=bool)
    X = X[:, keep_mask]
    gene_chr_labels_filtered = gene_chr_labels[keep_mask]

    # --- Step 2: normalize by seq depth ---
    X = normalize_by_seq_depth(X)

    # --- Step 3: log2(x+1) ---
    X = log2_plus1(X)

    # --- Step 4: subtract reference (1st pass) ---
    ref_groups_for_subtract = {k: v for k, v in reference_indices.items()}
    X = subtract_reference(
        X,
        ref_groups=ref_groups_for_subtract,
        use_bounds=config.ref_subtract_use_mean_bounds,
    )

    # --- Step 5: max-centered threshold ---
    X = apply_max_centered_threshold(X, threshold=config.max_centered_threshold)

    # --- Step 6: smooth per fake chromosome ---
    unique_chrs = list(dict.fromkeys(gene_chr_labels_filtered))
    for cname in unique_chrs:
        col_mask = gene_chr_labels_filtered == cname
        col_idx = np.where(col_mask)[0]
        if col_idx.shape[0] < 2:
            continue
        chr_block = X[:, col_idx]
        X[:, col_idx] = smooth_pyramidinal(chr_block, window_length=config.window_length)

    # --- Step 7: center cells (median) ---
    X = center_cells(X, method="median")

    # --- Step 8: subtract reference (2nd pass) ---
    X = subtract_reference(
        X,
        ref_groups=ref_groups_for_subtract,
        use_bounds=config.ref_subtract_use_mean_bounds,
    )

    return X.astype(np.float32), gene_chr_labels_filtered


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
    # Collect all observation cell rows
    obs_idx = np.concatenate([
        np.asarray(v, dtype=np.intp) for v in observation_indices.values()
    ])
    spike_expr = hspike_log2fc[obs_idx, :]  # (n_obs_cells, n_hspike_genes)

    # Build CNV -> pooled expression values map (mirrors R .get_gene_expr_by_cnv)
    cnv_to_vals: dict[float, list[NDArray[np.float64]]] = {}
    for cname, ccnv in HSPIKE_CHR_INFO:
        col_mask = gene_chr_labels == cname
        if not col_mask.any():
            continue
        # spike_expr[:, col_mask] is (n_obs, n_genes_in_chr), flatten
        vals = spike_expr[:, col_mask].ravel().astype(np.float64)
        if ccnv not in cnv_to_vals:
            cnv_to_vals[ccnv] = []
        cnv_to_vals[ccnv].append(vals)

    # Pool per CNV and compute mu, sigma
    state_mus = np.empty(6, dtype=np.float64)
    state_sigmas = np.empty(6, dtype=np.float64)

    for i, level in enumerate(I6_CNV_LEVELS_CALIBRATED):
        vals_list = cnv_to_vals.get(float(level), [])
        if vals_list:
            pooled = np.concatenate(vals_list)
        else:
            pooled = np.array([0.0])
        state_mus[i] = float(pooled.mean())
        state_sigmas[i] = float(pooled.std(ddof=1)) if pooled.shape[0] > 1 else 1.0

    return state_mus, state_sigmas


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
    obs_idx = np.concatenate([
        np.asarray(v, dtype=np.intp) for v in observation_indices.values()
    ])
    spike_expr = hspike_log2fc[obs_idx, :].astype(np.float64)

    # Build CNV -> pooled expression values map
    cnv_to_vals: dict[float, NDArray[np.float64]] = {}
    for cname, ccnv in HSPIKE_CHR_INFO:
        col_mask = gene_chr_labels == cname
        if not col_mask.any():
            continue
        vals = spike_expr[:, col_mask].ravel()
        if ccnv not in cnv_to_vals:
            cnv_to_vals[ccnv] = vals
        else:
            cnv_to_vals[ccnv] = np.concatenate([cnv_to_vals[ccnv], vals])

    n_cells_range = np.arange(1, max_num_cells + 1)
    log_n = np.log(n_cells_range.astype(np.float64))  # (max_num_cells,)

    slopes = np.empty(6, dtype=np.float64)
    intercepts = np.empty(6, dtype=np.float64)

    for i, level in enumerate(I6_CNV_LEVELS_CALIBRATED):
        expr_vals = cnv_to_vals.get(float(level), None)
        if expr_vals is None or expr_vals.shape[0] < 2:
            slopes[i] = -0.5
            intercepts[i] = 0.0
            continue

        # For each n, draw num_rounds replicates of size n, each replicate
        # is mean of n samples → sigma = sd of those num_rounds means
        sds = np.empty(max_num_cells, dtype=np.float64)
        for j, ncells in enumerate(n_cells_range):
            # Draw (ncells, num_rounds) samples
            samples = rng.choice(expr_vals, size=(ncells, num_rounds), replace=True)
            means_per_round = samples.mean(axis=0)  # (num_rounds,)
            sds[j] = means_per_round.std(ddof=1) if num_rounds > 1 else 0.0

        # Guard against zeros before log
        sds = np.maximum(sds, 1e-15)
        log_sd = np.log(sds)

        # OLS: log(sigma) = a * log(n) + b
        # X_mat = [log_n, 1], shape (max_num_cells, 2)
        X_mat = np.stack([log_n, np.ones_like(log_n)], axis=1)
        # Normal equations: (X'X) beta = X'y
        XtX = X_mat.T @ X_mat
        Xty = X_mat.T @ log_sd
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.array([-0.5, 0.0])
        slopes[i] = beta[0]
        intercepts[i] = beta[1]

    return slopes, intercepts


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

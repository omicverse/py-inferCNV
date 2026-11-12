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
    * a mean ``mu_s`` in linear-FC space
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
        ``invert_log2`` and outlier pruning) on the hspike counts. The
        result ``hspike_matrix`` has one row per simulated cell and one
        column per fake gene, so each hspike chromosome's contiguous gene
        columns carry the known CNV label of that block.
    4.  Computing ``mu_s = mean(hspike_matrix[chr == s_name])`` and
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
  ``(n_hspike_cells, n_hspike_genes)`` in linear-FC space.

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

Current parity notes
--------------------
The implementation intentionally keeps the stochastic hspike calibration
deterministic. Its default ``random_state=42`` mirrors the local R reference
fixture's ``set.seed(42)``. Python's PCG64 and R's Mersenne-Twister streams are
not bit-identical, so hspike remains empirical rather than bit-exact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray
import math

from pyinfercnv._rcompat.rbase import c_apply, r_mean, r_sd
from pyinfercnv._rcompat.rng import RRandom
from pyinfercnv._rcompat.spline import SmoothSpline
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


def _collapse_duplicate_x_by_mean(
    x_sorted: NDArray[np.float64],
    y_sorted: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Collapse exact duplicate x values by averaging y.

    R ``smooth.spline`` handles repeated ``x`` values internally rather than
    keeping an arbitrary first row. The mean aggregation is the least-surprising
    equal-weight reduction and keeps the downstream SciPy fit well-conditioned.
    Inputs must already be sorted by ``x``.
    """
    unique_x, first_idx, counts = np.unique(
        x_sorted, return_index=True, return_counts=True
    )
    y_sum = np.add.reduceat(y_sorted, first_idx)
    return unique_x.astype(np.float64), (y_sum / counts).astype(np.float64)


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
    hspike_matrix
        Optional, shape (n_hspike_cells, n_hspike_genes) float32 — the full
        post-Phase-1 hspike matrix retained for diagnostics/G3 verification.
        ``None`` when :func:`calibrate_i6_emission` is invoked with
        ``keep_matrix=False`` (default True during development).
    gene_chr_labels
        Optional, shape (n_hspike_genes,) of object dtype — fake-chr name per
        gene column of ``hspike_matrix``. Same ``None`` rule as above.

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
    hspike_matrix: NDArray[np.float32] | None = None
    gene_chr_labels: NDArray[np.object_] | None = None

    def sigmas_for_num_cells(self, num_cells: int) -> NDArray[np.float64]:
        """Return shape (6,) sd predicted for a subcluster of ``num_cells``.

        Mirror of R ``.get_state_emission_params`` (``inferCNV_HMM.R:586``).
        Uses the stored ``(sd_log_slope, sd_log_intercept)`` lm fit. Never
        returns negative or zero; callers should trust the result.
        """
        n = max(int(num_cells), 1)
        # R: exp(predict(lm, data.frame(num_cells = n))) = exp(b0 + b1 * log(n))
        return np.exp(self.sd_log_intercept + self.sd_log_slope * math.log(n)).astype(np.float64)


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
    expr_data_normalized: NDArray[np.float32],
    reference_groups: Mapping[str, NDArray[np.intp]] | None,
    *,
    observation_groups: Mapping[str, NDArray[np.intp]] | None = None,
    config: "InferCNVConfig | None" = None,
    sim_method: Literal["meanvar", "simple", "splatter"] = "meanvar",
    aggregate_normals: bool = False,
    num_cells_per_state: int = HSPIKE_DEFAULT_NUM_CELLS_PER_STATE,
    num_genes_per_chr: int = HSPIKE_DEFAULT_NUM_GENES_PER_CHR,
    trend_num_rounds: int = HSPIKE_TREND_NUM_ROUNDS,
    trend_max_num_cells: int = HSPIKE_TREND_MAX_NUM_CELLS,
    include_dropout: bool = True,
    keep_matrix: bool = False,
    random_state: int | None = 42,
    profile: dict[str, Any] | None = None,
    before_trend: Any = None,
) -> HspikeCalibration:
    """Top-level R-parity entry for i6 emission calibration.

    Executes the full ``.build_and_add_hspike`` → full-Phase-1-replay-through-
    step-16 → ``get_spike_dists`` → ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit``
    pipeline and returns a frozen :class:`HspikeCalibration`. Emission params
    live in **linear-FC space** (post-step14 invert_log2, post-step16
    outlier prune), matching R's step 17 HMM input.

    R-parity rewrite (2026-04-23):
      * D.1 — input is the post-step2-filter + post-step3-normalize full
        expr matrix (ALL cells: obs + ref), not ref-only raw counts
        (mirrors ``inferCNV_hidden_spike.R:59`` reading
        ``infercnv_obj@expr.data[,normal_cells_idx]``).
      * D.2 — mean/variance and dropout fits use **obs + ref groups
        pooled**, matching ``inferCNV_meanVarSim.R:178-186``
        ``.get_mean_var_table``.
      * D.3 — no extra hspike gene filter (R never refilters hspike).
      * D.4 — hspike is normalized once at build-time to
        ``median(colSums(normal_cells_expr))`` of the last normal group
        (``inferCNV_hidden_spike.R:160``).
      * D.5 — Phase 1 replay on hspike continues through step 14
        ``invert_log2`` and step 16 ``remove_outliers_norm``, so the
        calibration pool is linear-FC (``inferCNV_ops.R:2820``,
        ``inferCNV_ops.R:1987``).
      * D.6 — per-CN pooling uses observation cells only
        (``inferCNV_HMM.R:49-52``).
      * D.7 — fixed CN state order ``[0.01, 0.5, 1, 1.5, 2, 3]``; never
        reorder by empirical mean.

    Parameters
    ----------
    expr_data_normalized
        Shape (n_all_cells, n_genes_post_filter) dense float32 — post-filter,
        post-normalize, pre-log2 matrix **for ALL cells** (observation +
        reference). Consumed from :attr:`InferCNVResult.cpm_matrix_f32`.
    reference_groups
        Keys: reference group names. Values: integer arrays of **global**
        cell-row indices into ``expr_data_normalized`` (NOT ref-local
        indices; this changed from the previous contract). ``None`` (or
        empty) triggers R's reference-less branch
        (``inferCNV_hidden_spike.R:19-26``): the observation cells are
        used as fake normals.
    observation_groups
        Keys: observation (tumour) group names. Values: global cell indices.
        Used for mean/variance and dropout trend fitting (R pools both
        ``observation_grouped_cell_indices`` and
        ``reference_grouped_cell_indices`` at ``inferCNV_meanVarSim.R:180``).
        May be ``None`` or empty when ``reference_groups`` is populated; at
        least one of the two must be non-empty.
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
        If ``True``, retain ``hspike_matrix`` + ``gene_chr_labels`` on the
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
        When ``expr_data_normalized`` is empty, no reference or observation
        groups are supplied, group indices go out of bounds, or
        ``num_cells_per_state < 2`` (need ≥2 cells to compute per-group
        ``gene_means``).
    """
    # --- edge-case validation (G2 Q7) ---
    if sim_method in ("simple", "splatter"):
        raise NotImplementedError(
            f"sim_method={sim_method!r} is not implemented in Phase 2. "
            "Only 'meanvar' is ported. 'simple' and 'splatter' require "
            "inferCNV_simple_sim.R / SplatterScrape.R (~500 lines) and are deferred."
        )

    expr_full = np.asarray(expr_data_normalized, dtype=np.float64)
    if expr_full.size == 0 or expr_full.shape[0] == 0:
        raise ValueError(
            "expr_data_normalized is empty (0 cells). Cannot calibrate hspike emission."
        )
    n_all_cells, n_real_genes = expr_full.shape

    if num_cells_per_state < 2:
        raise ValueError(
            f"num_cells_per_state must be >= 2, got {num_cells_per_state}."
        )

    # Normalise inputs ---------------------------------------------------#
    def _copy_groups(src: Mapping[str, NDArray[np.intp]] | None) -> dict[str, NDArray[np.intp]]:
        if src is None:
            return {}
        out: dict[str, NDArray[np.intp]] = {}
        for k, v in src.items():
            arr = np.asarray(v, dtype=np.intp)
            if arr.size == 0:
                continue
            if arr.min() < 0 or arr.max() >= n_all_cells:
                raise ValueError(
                    f"Group {k!r} contains indices out of bounds for "
                    f"expr_data_normalized shape {expr_full.shape}"
                )
            out[k] = arr
        return out

    ref_groups_in = _copy_groups(reference_groups)
    obs_groups_in = _copy_groups(observation_groups)

    # R's .build_and_add_hspike branch (inferCNV_hidden_spike.R:9-26):
    #   has_reference_cells(obj): use reference_grouped_cell_indices as normals
    #   else: use observation_grouped_cell_indices as normals (proxy)
    if ref_groups_in:
        if aggregate_normals:
            all_idx = np.concatenate(list(ref_groups_in.values()))
            normal_cells_idx_lists: dict[str, NDArray[np.intp]] = {"normalsToUse": all_idx}
        else:
            normal_cells_idx_lists = dict(ref_groups_in)
    elif obs_groups_in:
        # ref-less fallback (R line 18-26)
        all_obs = np.concatenate(list(obs_groups_in.values()))
        normal_cells_idx_lists = {"normalsToUse": all_obs}
    else:
        raise ValueError(
            "calibrate_i6_emission requires at least one of reference_groups "
            "or observation_groups to contain cells."
        )

    # Validate each normal group has >=2 cells (G2 Q7)
    for grp_name, idx_arr in normal_cells_idx_lists.items():
        if idx_arr.shape[0] < 2:
            raise ValueError(
                f"Normal group {grp_name!r} has {idx_arr.shape[0]} cell(s); "
                "need >= 2 to compute gene means."
            )

    # Groupings used to fit mean/variance and dropout trends — R pools obs + ref
    # (inferCNV_meanVarSim.R:180). In the ref-less branch R uses the synthesised
    # obs→obs aggregation (infercnv_obj_tmp) which amounts to the same group.
    fit_groups: dict[str, NDArray[np.intp]] = {}
    fit_groups.update(obs_groups_in)
    fit_groups.update(ref_groups_in)
    if not fit_groups:
        # Degenerate (only synthesised normals from ref-less): fit on them
        fit_groups = dict(normal_cells_idx_lists)

    from pyinfercnv.config import InferCNVConfig as _Config
    cfg = config if config is not None else _Config()

    # R's generator: ``random_state=s`` is ``set.seed(s)`` before ``infercnv::run`` (inferCNV
    # itself never sets a seed; the hspike is the first random step).
    rng = random_state if isinstance(random_state, RRandom) else RRandom(
        42 if random_state is None else int(random_state))

    # ------------------------------------------------------------------ #
    # Stage 1: build hspike counts (D.1 + D.2 + D.4)                      #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    hspike_counts, gene_chr_labels_full, reference_indices, observation_indices = (
        _build_hspike_counts(
            expr_full,
            normal_cells_idx_lists=normal_cells_idx_lists,
            fit_groups=fit_groups,
            num_cells_per_state=num_cells_per_state,
            num_genes_per_chr=num_genes_per_chr,
            sim_method=sim_method,
            include_dropout=include_dropout,
            rng=rng,
        )
    )

    _record_profile(profile, "hspike_sim", t0, rss0)

    # ------------------------------------------------------------------ #
    # Stage 2: Phase 1 replay through steps 4-16 (D.3 no filter, D.5 + prune) #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    hspike_linear, gene_chr_labels_final = _run_phase1_replay_on_hspike(
        hspike_counts,
        gene_chr_labels=gene_chr_labels_full,
        reference_indices=reference_indices,
        config=cfg,
    )

    _record_profile(profile, "hspike_phase1_replay", t0, rss0)

    # ------------------------------------------------------------------ #
    # Stage 3: get per-CNV mu/sigma in linear FC (obs-only pooling, D.6)  #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    state_mus, state_sigmas = _gene_expr_mean_sd_by_cnv(
        hspike_linear,
        gene_chr_labels_final,
        observation_indices,
    )

    _record_profile(profile, "hspike_get_dists", t0, rss0)

    # R draws the trend's samples at the HMM step, after the subclustering (of the data and of
    # the hspike) has used the generator; ``before_trend(rng, hspike, ref_idx, obs_idx)`` runs
    # those steps on the same generator.
    if before_trend is not None:
        before_trend(rng, hspike_linear, reference_indices, observation_indices)

    # ------------------------------------------------------------------ #
    # Stage 4: trend LM fit (lm(log(sd_n) ~ log(n)) in linear FC)         #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    sd_log_slope, sd_log_intercept = _fit_cnv_sd_vs_num_cells_trend(
        hspike_linear,
        gene_chr_labels_final,
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
        hspike_matrix=hspike_linear if keep_matrix else None,  # field name kept; semantics now linear-FC
        gene_chr_labels=(
            gene_chr_labels_final if keep_matrix else None
        ),
    )


# --------------------------------------------------------------------------- #
# Private helpers (stable signatures for G2 review; bodies in Wave 2)         #
# --------------------------------------------------------------------------- #


def _row_means_r(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``rowMeans`` of a genes x cells matrix given here as cells x genes: long-double sums."""
    return (np.sum(np.asarray(m, dtype=np.longdouble), axis=0) / m.shape[0]).astype(np.float64)


def _vars_r(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``apply(x, 1, var)`` per gene (columns here): cov.c two-pass long-double sums."""
    m = np.asarray(m, dtype=np.float64)
    n = m.shape[0]
    if n < 2:
        return np.full(m.shape[1], np.nan)
    ld = m.astype(np.longdouble)
    tmp = np.sum(ld, axis=0) / n
    tmp = tmp + np.sum(ld - tmp, axis=0) / n
    xm = tmp.astype(np.float64)
    d = m - xm
    return (np.sum((d * d).astype(np.longdouble), axis=0) / (n - 1)).astype(np.float64)


def _mean_r_cols(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``mean`` of each column (two-pass, long double)."""
    ld = np.asarray(m, dtype=np.longdouble)
    n = m.shape[0]
    s = np.sum(ld, axis=0) / n
    s = s + np.sum(ld - s, axis=0) / n
    return s.astype(np.float64)


def _build_hspike_counts(
    expr_data_normalized: NDArray[np.float64],
    *,
    normal_cells_idx_lists: Mapping[str, NDArray[np.intp]],
    fit_groups: Mapping[str, NDArray[np.intp]],
    num_cells_per_state: int,
    num_genes_per_chr: int,
    sim_method: Literal["meanvar", "simple", "splatter"],
    include_dropout: bool,
    rng: RRandom,
) -> tuple[NDArray[np.float64], NDArray[np.object_], dict[str, NDArray[np.intp]], dict[str, NDArray[np.intp]]]:
    """Build the synthetic hspike counts as R ``.build_and_add_hspike`` (``sim_method="meanvar"``).

    The random draws follow R's order on R's generator: ``sample`` of the source genes, then per
    normal group the normal and the spiked cells, each drawn cell by cell (``rnorm`` per gene,
    skipped when the predicted variance is 0) followed by the dropout (``runif`` per gene and
    cell, gene by gene). Mean/variance and dropout trends are R ``smooth.spline`` fits (GCV).

    Returns ``(counts, gene_chr_labels, reference_indices, observation_indices)``; ``counts`` is
    cells x genes, already normalised to the median library size of the last normal group.
    """
    if sim_method in ("simple", "splatter"):
        raise NotImplementedError(
            f"sim_method={sim_method!r} is not implemented. Only 'meanvar' is supported."
        )

    expr = np.asarray(expr_data_normalized, dtype=np.float64)
    n_real_genes = expr.shape[1]

    # Fake chromosome layout (R .get_hspike_chr_info)
    n_remaining = n_real_genes - 10 * num_genes_per_chr
    if n_remaining < num_genes_per_chr:
        n_remaining = num_genes_per_chr
    chr_ngenes = [n_remaining if cname == "chr_F" else num_genes_per_chr for cname, _ in HSPIKE_CHR_INFO]
    chr_names = [cname for cname, _ in HSPIKE_CHR_INFO]
    chr_cnvs = [ccnv for _, ccnv in HSPIKE_CHR_INFO]
    n_fake_genes = sum(chr_ngenes)
    gene_chr_labels = np.repeat(np.asarray(chr_names, dtype=object), chr_ngenes)
    cnv_per_gene = np.repeat(np.asarray(chr_cnvs, dtype=np.float64), chr_ngenes)

    # R: sample(x = seq_len(nrow(expr)), size = num_genes, replace = TRUE)
    genes_means_use_idx = rng.sample(n_real_genes, n_fake_genes, replace=True)

    # The trends are refitted inside every simulation call in R, on the same data, so one fit
    # serves all calls.
    mean_var_spline = _fit_meanvar_spline(expr, fit_groups)
    dropout_spline = _fit_dropout_logistic_params(expr, fit_groups) if include_dropout else None

    cells_list: list[NDArray[np.float64]] = []
    reference_indices: dict[str, NDArray[np.intp]] = {}
    observation_indices: dict[str, NDArray[np.intp]] = {}
    cell_counter = 0
    last_normal_cells_expr: NDArray[np.float64] | None = None   # R variable-scope leak (line 160)

    for normal_type, cell_idx in normal_cells_idx_lists.items():
        normal_cells_expr = expr[np.asarray(cell_idx, dtype=np.intp), :]
        last_normal_cells_expr = normal_cells_expr
        gene_means = _row_means_r(normal_cells_expr)[genes_means_use_idx]
        gene_means[gene_means == 0] = 0.001

        sim_normal = _simulate_meanvar_counts(gene_means, mean_var_spline, num_cells_per_state,
                                              dropout_spline, rng)
        hspike_gene_means = np.where(cnv_per_gene != 1.0, gene_means * cnv_per_gene, gene_means)
        sim_tumor = _simulate_meanvar_counts(hspike_gene_means, mean_var_spline, num_cells_per_state,
                                             dropout_spline, rng)

        reference_indices[f"simnorm_cell_{normal_type}"] = np.arange(
            cell_counter, cell_counter + num_cells_per_state, dtype=np.intp)
        cell_counter += num_cells_per_state
        observation_indices[f"spike_tumor_cell_{normal_type}"] = np.arange(
            cell_counter, cell_counter + num_cells_per_state, dtype=np.intp)
        cell_counter += num_cells_per_state
        cells_list.append(sim_normal)
        cells_list.append(sim_tumor)

    counts = np.concatenate(cells_list, axis=0)

    # R: normalize_counts_by_seq_depth(.hspike, median(colSums(normal_cells_expr)))
    if last_normal_cells_expr is not None:
        libsizes = np.sum(last_normal_cells_expr.astype(np.longdouble), axis=1).astype(np.float64)
        target_libsize = float(np.median(libsizes))
        if target_libsize > 0:
            from pyinfercnv.preprocess import normalize_by_seq_depth  # late import
            counts = normalize_by_seq_depth(counts, normalize_factor=target_libsize)

    return counts, gene_chr_labels, reference_indices, observation_indices


def _fit_dropout_logistic_params(
    counts: NDArray[np.float64],
    cell_groupings: Mapping[str, NDArray[np.intp]] | None = None,
) -> SmoothSpline:
    """R ``.get_logistic_params``' spline: ``smooth.spline(log(m), p0)`` over genes with
    ``m > 0``, (m, p0) pooled over the observation then reference groups.

    (R also fits a logistic curve with ``nls``; only the spline is used for the dropout.)"""
    counts_f = np.asarray(counts, dtype=np.float64)
    groups = (cell_groupings if cell_groupings is not None
              else {"allcells": np.arange(counts_f.shape[0], dtype=np.intp)})
    m_list, p0_list = [], []
    for idx in groups.values():
        sub = counts_f[np.asarray(idx, dtype=np.intp), :]
        m_list.append(_row_means_r(sub))
        p0_list.append((sub == 0).sum(axis=0) / sub.shape[0])
    m = np.concatenate(m_list)
    p0 = np.concatenate(p0_list)
    keep = m > 0
    return SmoothSpline(c_apply(math.log, m[keep]), p0[keep], spar=None)


def _fit_meanvar_spline(
    counts: NDArray[np.float64],
    cell_groupings: Mapping[str, NDArray[np.intp]] | None,
) -> SmoothSpline:
    """R ``smooth.spline(log(v + 1) ~ log(m + 1))`` over genes x groups (``.get_mean_var_table``:
    observation then reference groups, gene mean and variance per group)."""
    counts_f = np.asarray(counts, dtype=np.float64)
    groups = (cell_groupings if cell_groupings is not None
              else {"allcells": np.arange(counts_f.shape[0], dtype=np.intp)})
    m_all, v_all = [], []
    for grp_idx in groups.values():
        sub = counts_f[np.asarray(grp_idx, dtype=np.intp), :]
        m_all.append(_row_means_r(sub))
        v_all.append(_vars_r(sub))
    m = np.concatenate(m_all)
    v = np.concatenate(v_all)
    return SmoothSpline(c_apply(math.log, m + 1.0), c_apply(math.log, v + 1.0), spar=None)


def _simulate_meanvar_counts(
    gene_means: NDArray[np.float64],
    mean_var_spline: SmoothSpline,
    num_cells: int,
    dropout_spline: SmoothSpline | None,
    rng: RRandom,
) -> NDArray[np.float64]:
    """R ``.get_simulated_cell_matrix_using_meanvar_trend_helper``: ``(num_cells, n_genes)``.

    Per gene ``var = max(exp(spline(log(m + 1))) - 1, 0)`` and per cell ``round(max(rnorm(1, m,
    sqrt(var)), 0))``, drawn cell by cell; then ``.apply_dropout``."""
    gene_means = np.asarray(gene_means, dtype=np.float64)
    pred = mean_var_spline.predict(c_apply(math.log, gene_means + 1.0))
    var = np.maximum(c_apply(math.exp, pred) - 1.0, 0.0)
    sd = c_apply(math.sqrt, var)
    vals = np.zeros((num_cells, gene_means.size))
    pos = gene_means > 0
    if pos.any():
        mu = np.broadcast_to(gene_means[pos], (num_cells, int(pos.sum()))).ravel()
        sg = np.broadcast_to(sd[pos], (num_cells, int(pos.sum()))).ravel()
        draws = rng.rnorm(mu, sg).reshape(num_cells, -1)
        vals[:, pos] = np.rint(np.maximum(draws, 0.0))
    if dropout_spline is not None:
        vals = _apply_dropout(vals, dropout_spline, rng)
    return vals


def _apply_dropout(
    counts: NDArray[np.float64],
    dropout_spline: SmoothSpline,
    rng: RRandom,
) -> NDArray[np.float64]:
    """R ``.apply_dropout``: per gene, ``padj = max((p_drop * n - nzeros) / nremaining, 0)`` with
    ``p_drop`` the dropout spline at ``log(mean)``; each entry is zeroed when ``runif(1) <= padj``,
    one uniform per entry, gene by gene."""
    x = np.asarray(counts, dtype=np.float64)          # cells x genes
    ntotal = x.shape[0]
    mean_val = _mean_r_cols(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.where(mean_val > 0, c_apply(math.log, np.where(mean_val > 0, mean_val, 1.0)), -np.inf)
        p_drop = dropout_spline.predict(logm)
        nzeros = (x == 0).sum(axis=0).astype(np.float64)
        nremaining = ntotal - nzeros
        padj = (p_drop * ntotal - nzeros) / nremaining
    padj = np.where(np.isnan(padj), 0.0, np.maximum(padj, 0.0))
    u = rng.unif_rand(x.size).reshape(x.shape[1], ntotal).T     # gene-major draws
    return np.where(u <= padj[None, :], 0.0, x)


def _run_phase1_replay_on_hspike(
    hspike_counts: NDArray[np.float64],
    *,
    gene_chr_labels: NDArray[np.object_],
    reference_indices: dict[str, NDArray[np.intp]],
    config: "InferCNVConfig",
) -> tuple[NDArray[np.float64], NDArray[np.object_]]:
    """The steps R's ``run()`` also applies to ``infercnv_obj@.hspike`` up to the HMM.

    log2(x + 1), reference subtraction, max-centred threshold, per-chromosome smoothing,
    per-cell centring, second reference subtraction and ``invert_log2``; outliers are removed
    only with ``prune_outliers=True``, as in R. No gene filter or normalisation (the hspike is
    created after those steps)."""
    from pyinfercnv.preprocess import (  # noqa: PLC0415
        apply_max_centered_threshold,
        invert_log2,
        log2_plus1,
        subtract_reference,
    )
    from pyinfercnv.smooth import smooth_pyramidinal  # noqa: PLC0415
    from pyinfercnv.center import center_cells  # noqa: PLC0415
    from pyinfercnv.cna import prune_outliers  # noqa: PLC0415

    X = log2_plus1(np.asarray(hspike_counts, dtype=np.float64))
    refs = dict(reference_indices)
    X = subtract_reference(X, ref_groups=refs, use_bounds=config.ref_subtract_use_mean_bounds)
    X = apply_max_centered_threshold(X, threshold=config.max_centered_threshold)
    for cname in dict.fromkeys(gene_chr_labels):
        col_idx = np.where(gene_chr_labels == cname)[0]
        if col_idx.shape[0] >= 2:
            X[:, col_idx] = smooth_pyramidinal(X[:, col_idx], window_length=config.window_length)
    X = center_cells(X, method="median")
    X = subtract_reference(X, ref_groups=refs, use_bounds=config.ref_subtract_use_mean_bounds)
    X = invert_log2(X)
    if config.prune_outliers:
        X = prune_outliers(
            X,
            method=config.outlier_method_bound,
            lower_bound=config.outlier_lower_bound,
            upper_bound=config.outlier_upper_bound,
        )
    return X, gene_chr_labels


def _pooled_by_cnv(
    hspike_linear: NDArray[np.float64],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
) -> dict[float, NDArray[np.float64]]:
    """R ``.get_gene_expr_by_cnv``: per CNV level (in order of first appearance), the spiked
    cells' values of its fake chromosomes, genes within cells (R ``c(matrix)``)."""
    obs_idx = np.concatenate([np.asarray(v, dtype=np.intp) for v in observation_indices.values()])
    spike = np.asarray(hspike_linear, dtype=np.float64)[obs_idx, :]
    out: dict[float, list[NDArray[np.float64]]] = {}
    for cname, ccnv in HSPIKE_CHR_INFO:
        cols = gene_chr_labels == cname
        if cols.any():
            out.setdefault(float(ccnv), []).append(spike[:, cols].ravel())
    return {k: np.concatenate(v) for k, v in out.items()}


def _gene_expr_mean_sd_by_cnv(
    hspike_linear: NDArray[np.float64],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R ``get_spike_dists``: ``mean`` and ``sd`` of each CNV level's pooled values, returned in
    the order of :data:`I6_CNV_LEVELS_CALIBRATED`."""
    pooled = _pooled_by_cnv(hspike_linear, gene_chr_labels, observation_indices)
    mus = np.array([r_mean(pooled[float(lv)]) for lv in I6_CNV_LEVELS_CALIBRATED])
    sds = np.array([r_sd(pooled[float(lv)]) for lv in I6_CNV_LEVELS_CALIBRATED])
    return mus, sds


def _fit_cnv_sd_vs_num_cells_trend(
    hspike_linear: NDArray[np.float64],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
    *,
    num_rounds: int,
    max_num_cells: int,
    rng: RRandom,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit``: per CNV level and ``n`` in
    1..``max_num_cells``, ``replicate(num_rounds, sample(vals, n, replace = TRUE))``, the sd of
    the ``rowMeans`` of that n x rounds matrix, and ``lm(log(sd) ~ log(n))`` (``n = 1`` gives NA).

    Levels are drawn in R's order (first appearance in the fake genome). In R this runs at the
    HMM step, after the subclustering has consumed random numbers, so these draws continue the
    hspike stream rather than reproduce R's; the fit averages ``num_rounds`` rounds."""
    pooled = _pooled_by_cnv(hspike_linear, gene_chr_labels, observation_indices)
    log_n = np.log(np.arange(2, max_num_cells + 1, dtype=np.float64))
    fits: dict[float, tuple[float, float]] = {}
    for level, vals in pooled.items():
        sds = []
        for ncells in range(1, max_num_cells + 1):
            idx = rng.sample(vals.size, num_rounds * ncells, replace=True)
            if ncells == 1:
                continue                                           # sd of one mean is NA
            m = vals[idx].reshape(num_rounds, ncells).T            # ncells x rounds (column = round)
            row_means = (np.sum(m.astype(np.longdouble), axis=1) / num_rounds).astype(np.float64)
            sds.append(r_sd(row_means))
        y = np.log(np.asarray(sds))
        X = np.column_stack([np.ones_like(log_n), log_n])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fits[level] = (float(beta[1]), float(beta[0]))
    slopes = np.array([fits[float(lv)][0] for lv in I6_CNV_LEVELS_CALIBRATED])
    intercepts = np.array([fits[float(lv)][1] for lv in I6_CNV_LEVELS_CALIBRATED])
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

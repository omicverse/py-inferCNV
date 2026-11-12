"""Hidden spike-in calibration of i6 emission means and subcluster SD trends.

The default ``r_compatible`` backend uses a local R MT/Inversion/Rejection
stream and weighted cubic smoothing splines with R's GCV objective. Inputs
are all post-filter, depth-normalized cells in R chromosome/start order,
with float64 precision. Reference and observation groups are sorted by name.
The pipeline supplies that order; direct callers must provide it explicitly.

Simulation mirrors inferCNV_hidden_spike.R and inferCNV_meanVarSim.R:
normal draws run cell-first, dropout draws gene-first (including zeros).
The synthetic genome is replayed through log2, reference subtraction,
clipping, smoothing, centering, second subtraction, inverse log2 and optional
outlier pruning, using the same configuration as the real matrix.

inferCNV_HMM.R pools observation values by CN level, then fits log(SD)
against log(cell count). Its rowMeans-across-rounds convention and n=1 RNG
consumption are preserved. The local stream continues from hspike simulation;
R's full run consumes additional random numbers during tree construction,
so the final trend need not be bit-identical to a historical R run. GCV
optimizer rounding can also cause rare simulated-count differences.

``numpy_legacy`` retains the prior PCG64, unweighted approximate spline and
sampling order for historical reconstruction. Neither implementation turns
CNV agreement into verified malignant-cell accuracy. The four-case numerical
validation is recorded in results/03_malignancy/07_hmm_calibration_20261112/.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline
from scipy.special import expit
from .r_compat import RCompatibleRNG, RSmoothingSpline, fit_r_smoothing_spline

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
        Shape (6,) float64 — per-state mean in linear-FC space, as read from
        the post-Phase-1 hspike matrix restricted to that state's fake-chr
        block. Aligns with :data:`pyinfercnv.hmm.i6.I6_DEFAULT_MUS` in shape
        and semantics but with R-calibrated values.
    state_sigmas
        Shape (6,) float64 — per-state sd at ``num_cells=1`` (i.e., the raw
        per-gene-value spread within the fake-chr block). For subcluster
        HMM calls, use :meth:`sigmas_for_num_cells` to apply the trend fit.
    sd_log_slope, sd_log_intercept
        Shape (6,) float64 each — per-state lm coefficients for
        ``log(sigma(n)) ~ a * log(n) + b``. The preserved R rowMeans
        convention averages across simulation rounds, so slopes are near
        zero and the intercept is not log(state_sigmas); see the fit below.
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
        Shape (n_all_cells, n_genes_post_filter), post-normalize and pre-log2,
        for all observation and reference cells. The R-compatible backend
        expects float64 values in R chromosome/start order. Consumed from
        the historical field :attr:`InferCNVResult.cpm_matrix_f32`.
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
        preprocess/smooth/center/subtract/threshold fields and
        ``hspike_calibration_backend`` are honoured.
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
        Seed for an isolated R-compatible MT stream (default backend), or
        PCG64 (legacy backend). ``None`` draws a nondeterministic seed.
        Reproducible production runs must provide an integer seed.
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

    from pyinfercnv.config import InferCNVConfig as _Config
    cfg = config if config is not None else _Config()
    backend = cfg.hspike_calibration_backend
    if backend not in ("r_compatible", "numpy_legacy"):
        raise ValueError(f"Unknown hspike_calibration_backend: {backend!r}")
    expr_full = np.asarray(expr_data_normalized, dtype=np.float64 if backend == "r_compatible" else np.float32)
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
    if backend == "r_compatible":
        ref_groups_in = dict(sorted(ref_groups_in.items()))
        obs_groups_in = dict(sorted(obs_groups_in.items()))

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

    rng = RCompatibleRNG(random_state) if backend == "r_compatible" else np.random.default_rng(random_state)

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


def _build_hspike_counts(
    expr_data_normalized: NDArray[np.float32],
    *,
    normal_cells_idx_lists: Mapping[str, NDArray[np.intp]],
    fit_groups: Mapping[str, NDArray[np.intp]],
    num_cells_per_state: int,
    num_genes_per_chr: int,
    sim_method: Literal["meanvar", "simple", "splatter"],
    include_dropout: bool,
    rng: np.random.Generator | RCompatibleRNG,
) -> tuple[NDArray[np.float32], NDArray[np.object_], dict[str, NDArray[np.intp]], dict[str, NDArray[np.intp]]]:
    """Build the synthetic hspike counts matrix mirroring R ``.build_and_add_hspike``.

    R-parity notes:
      * Gene means come from ``expr_data_normalized[normal_cells_idx, :]``
        — post-normalize, NOT raw counts (``inferCNV_hidden_spike.R:59-60``).
      * Mean/variance + dropout trend fits use ``fit_groups`` (typically
        obs + ref pooled per ``inferCNV_meanVarSim.R:180``).
      * At the end, hspike counts are re-normalized to
        ``median(colSums(last_normal_cells_expr))`` mirroring
        ``inferCNV_hidden_spike.R:160``. This uses the **last** group's
        libsize median (R variable-scope leak; preserved for parity).

    Returns
    -------
    counts
        Shape (n_hspike_cells, n_hspike_genes), already normalized; float64
        for the R-compatible backend and float32 for the legacy backend.
    gene_chr_labels
        Shape (n_hspike_genes,) object dtype — per-gene fake chromosome name.
    reference_indices, observation_indices
        Per-group row-index arrays into ``counts``.
    """
    if sim_method in ("simple", "splatter"):
        raise NotImplementedError(
            f"sim_method={sim_method!r} is not implemented. Only 'meanvar' is supported."
        )

    n_real_genes = expr_data_normalized.shape[1]

    # Build fake chromosome layout (R .get_hspike_chr_info)
    n_remaining = n_real_genes - 10 * num_genes_per_chr
    if n_remaining < num_genes_per_chr:
        n_remaining = num_genes_per_chr

    chr_ngenes: list[int] = []
    chr_names: list[str] = []
    chr_cnvs: list[float] = []
    for cname, ccnv in HSPIKE_CHR_INFO:
        n_g = n_remaining if cname == "chr_F" else num_genes_per_chr
        chr_ngenes.append(n_g)
        chr_names.append(cname)
        chr_cnvs.append(ccnv)

    n_fake_genes = sum(chr_ngenes)

    gene_chr_labels = np.empty(n_fake_genes, dtype=object)
    offset = 0
    for cname, n_g in zip(chr_names, chr_ngenes):
        gene_chr_labels[offset: offset + n_g] = cname
        offset += n_g

    # Random gene-index sample (R line 44: sample(seq_len(nrow), size=num_genes, replace=TRUE))
    genes_means_use_idx = rng.choice(n_real_genes, size=n_fake_genes, replace=True)

    # D.2 — fit mean-var spline + dropout on the pooled obs+ref groups
    spline_method = "r_compatible" if isinstance(rng, RCompatibleRNG) else "numpy_legacy"
    spline_knots = _fit_meanvar_spline(expr_data_normalized, fit_groups, method=spline_method)
    dropout_logistic_params: RSmoothingSpline | tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if include_dropout:
        dropout_logistic_params = _fit_dropout_logistic_params(
            expr_data_normalized, fit_groups, method=spline_method
        )

    # Build cells: per normal group, normals first then spiked-tumour (R cbind order)
    cells_list: list[NDArray[np.float32]] = []
    reference_indices: dict[str, NDArray[np.intp]] = {}
    observation_indices: dict[str, NDArray[np.intp]] = {}
    cell_counter = 0
    last_normal_cells_expr: NDArray[np.float32] | None = None  # R scope-leak at line 160

    for normal_type, cell_idx in normal_cells_idx_lists.items():
        cell_idx_arr = np.asarray(cell_idx, dtype=np.intp)
        normal_cells_expr = expr_data_normalized[cell_idx_arr, :]  # (n_cells, n_real_genes)
        last_normal_cells_expr = normal_cells_expr

        # Per-gene means from this group (R line 60)
        gene_means_orig = normal_cells_expr.mean(axis=0).astype(np.float64)
        gene_means = gene_means_orig[genes_means_use_idx].copy()
        gene_means[gene_means == 0] = 1e-3  # R line 65

        sim_normal = _simulate_meanvar_counts(
            gene_means, spline_knots, num_cells_per_state,
            dropout_logistic_params, rng,
        )

        # CN-scaled gene_means (R line 103-111)
        hspike_gene_means = gene_means.copy()
        offset = 0
        for cname, n_g, ccnv in zip(chr_names, chr_ngenes, chr_cnvs):
            if ccnv != 1.0:
                hspike_gene_means[offset: offset + n_g] *= ccnv
            offset += n_g

        sim_tumor = _simulate_meanvar_counts(
            hspike_gene_means, spline_knots, num_cells_per_state,
            dropout_logistic_params, rng,
        )

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

    # D.4 — normalize hspike to the real normals' median libsize (R line 160).
    # The semantic quirk: R's `normal_cells_expr` at line 160 is the LAST
    # iteration's binding from the for-loop above (R variable scope). We
    # preserve that behaviour for parity.
    if last_normal_cells_expr is not None:
        target_libsize = float(np.median(last_normal_cells_expr.sum(axis=1)))
        if target_libsize > 0:
            from pyinfercnv.preprocess import normalize_by_seq_depth  # late import
            counts = normalize_by_seq_depth(
                counts, normalize_factor=target_libsize
            ).astype(np.float64 if isinstance(rng, RCompatibleRNG) else np.float32)

    return counts, gene_chr_labels, reference_indices, observation_indices


def _fit_dropout_logistic_params(
    counts: NDArray[np.float32],
    cell_groupings: Mapping[str, NDArray[np.intp]] | None = None,
    *, method: str = "r_compatible",
) -> RSmoothingSpline | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit dropout probability vs log(mean) from the reference matrix.

    Returns a callable R-compatible spline, or legacy interpolation grids.
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

    # R `.get_logistic_params` filters zero means, then fits
    # `smooth.spline(log(m), p0)` (inferCNV_simple_sim.R:188-219). This is
    # intentionally different from the mean-var spline, which uses log(m+1).
    positive = m > 0.0
    if not np.any(positive):
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    log_m = np.log(m[positive])
    p0 = p0[positive]
    if method == "r_compatible":
        return fit_r_smoothing_spline(log_m, p0)
    order = np.argsort(log_m)
    log_m_sorted = log_m[order]
    p0_sorted = np.clip(p0[order], 0.0, 1.0)

    x_u, y_u = _collapse_duplicate_x_by_mean(log_m_sorted, p0_sorted)

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
    *, method: str = "r_compatible",
) -> RSmoothingSpline | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit R ``smooth.spline(log(v+1) ~ log(m+1))`` gene-wise mean-variance trend.

    Returns a callable weighted GCV spline by default. The legacy backend
    returns grid values from a fixed-residual-budget UnivariateSpline.

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
    if method == "r_compatible":
        return fit_r_smoothing_spline(log_m, log_v)

    order = np.argsort(log_m)
    log_m_sorted = log_m[order]
    log_v_sorted = log_v[order]

    x_u, y_u = _collapse_duplicate_x_by_mean(log_m_sorted, log_v_sorted)

    if x_u.shape[0] < 4:
        # Degenerate: return trivial flat spline
        return x_u.astype(np.float64), y_u.astype(np.float64)

    # Approximate fit: s=len(x_u) is not R smooth.spline GCV.
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
    spline_knots: RSmoothingSpline | tuple[NDArray[np.float64], NDArray[np.float64]],
    num_cells: int,
    dropout_logistic_params: RSmoothingSpline | tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    rng: np.random.Generator | RCompatibleRNG,
) -> NDArray[np.float32]:
    """Sample ``(num_cells, n_genes)`` counts from the mean-var trend.

    Per-gene: ``var = max(exp(spline(log(m+1))) - 1, 0)``, then draw
    ``round(max(N(m, sqrt(var)), 0))`` (R
    ``inferCNV_meanVarSim.R:105-119``). Optionally apply logistic dropout
    via ``_apply_dropout`` if ``dropout_logistic_params`` is not None.

    Returns
    -------
    simulated
        Shape ``(num_cells, len(gene_means))``; float64 for the R-compatible
        backend and float32 for the legacy backend.
    """
    gene_means_f = np.asarray(gene_means, dtype=np.float64)
    n_genes = gene_means_f.shape[0]

    # Vectorized: predict log_var for all genes at once
    log_m_genes = np.log(gene_means_f + 1.0)
    pred_log_v = (spline_knots(log_m_genes) if isinstance(spline_knots, RSmoothingSpline)
                  else np.interp(log_m_genes, *spline_knots))
    gene_var = np.maximum(np.exp(pred_log_v) - 1.0, 0.0)  # (n_genes,)
    gene_sd = np.sqrt(gene_var)  # (n_genes,)

    # Draw normal random matrix in one shot
    mu = gene_means_f[None, :]   # (1, n_genes)
    sd = gene_sd[None, :]        # (1, n_genes)
    if isinstance(rng, RCompatibleRNG):
        # R loops over cells, then genes, calling rnorm(1) at each entry.
        # Its subsequent dropout loop has the opposite (gene-first) order.
        active = gene_sd > 0
        z = np.zeros((num_cells, n_genes))
        z[:, active] = rng.standard_normal((num_cells, int(active.sum())))
        x = mu + sd * z
    else:
        x = rng.normal(loc=mu, scale=sd, size=(num_cells, n_genes))
    x = np.maximum(x, 0.0)
    x = np.round(x).astype(np.float64 if isinstance(rng, RCompatibleRNG) else np.float32)

    if dropout_logistic_params is not None:
        x = _apply_dropout(x, dropout_logistic_params, rng)

    return x


def _apply_dropout(
    counts: NDArray[np.float32],
    dropout_logistic_params: RSmoothingSpline | tuple[NDArray[np.float64], NDArray[np.float64]],
    rng: np.random.Generator | RCompatibleRNG,
) -> NDArray[np.float32]:
    """Logistic-dropout post-processing, mirror of R ``.apply_dropout``.

    R reference: ``inferCNV_meanVarSim.R:122-161``. The ``padj`` adjustment
    line 137 must be preserved verbatim for R-parity.
    """
    native_r = isinstance(rng, RCompatibleRNG)
    counts_f = np.asarray(counts, dtype=np.float64 if native_r else np.float32)
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
    dropout_prob = (dropout_logistic_params(log_mean_per_gene)
                    if isinstance(dropout_logistic_params, RSmoothingSpline)
                    else np.interp(log_mean_per_gene, *dropout_logistic_params))

    # R padj formula (line 137, preserved verbatim):
    # padj = ( (dropout_prob * ntotal) - nzeros ) / nremaining
    # padj = max(padj, 0)
    padj = (dropout_prob * ntotal - nzeros_per_gene) / np.maximum(nremaining_per_gene, 1.0)
    padj = np.maximum(padj, 0.0)  # (n_genes,)

    # Draw for every entry; zero entries remain zero after the coin flip.
    u = rng.random(counts_f.T.shape).T if native_r else rng.random(counts_f.shape)
    drop_mask = ((u <= padj[None, :]) if native_r else (u < padj[None, :])) & (counts_f != 0)
    out = np.where(drop_mask, 0.0, counts_f).astype(counts_f.dtype)
    return out


def _run_phase1_replay_on_hspike(
    hspike_counts: NDArray[np.float32],
    *,
    gene_chr_labels: NDArray[np.object_],
    reference_indices: dict[str, NDArray[np.intp]],
    config: "InferCNVConfig",
) -> tuple[NDArray[np.float32], NDArray[np.object_]]:
    """Re-run steps 4-16 of Phase 1 on the already-normalized hspike counts.

    R-parity (D.3, D.5):
      * **No initial gene filter**. R's ``.build_and_add_hspike`` never
        refilters hspike; step 2 ``require_above_min_mean_expr_cutoff`` is
        only applied to the main matrix.
      * **No initial normalize**. The hspike was already normalized by
        :func:`_build_hspike_counts` to the real normals' libsize target
        (``inferCNV_hidden_spike.R:160``); step 3 in the main pipeline is
        not mirrored to .hspike because .hspike didn't exist at that point.
      * Mirrors step 4 log2 onwards (``inferCNV_ops.R`` ``log2xplus1``
        mirror at line ~960; ``subtract_ref_expr_from_obs`` mirror at
        line 1697; smooth mirror at line 2429; center mirror at line 2083;
        second subtract_ref mirror at line 1697; **invert_log2 mirror at
        line 2820**; **remove_outliers_norm mirror at line 1987**).

    Steps applied (in order):
        1. ``log2_plus1``
        2. ``subtract_reference`` (pass 1, uses ``reference_indices``)
        3. ``apply_max_centered_threshold``
        4. ``smooth_pyramidinal`` per fake chromosome
        5. ``center_cells`` (median)
        6. ``subtract_reference`` (pass 2)
        7. ``invert_log2`` — ``2 ** x``  (R step 14)
        8. ``prune_outliers`` in linear FC only when enabled (R step 16)

    Returns
    -------
    hspike_linear
        Shape ``(n_hspike_cells, n_hspike_genes)`` float32, in linear FC
        (post-invert_log2, post-outlier-prune).
    gene_chr_labels
        Shape ``(n_hspike_genes,)`` — unchanged (no filter applied).
    """
    from pyinfercnv.preprocess import (  # noqa: PLC0415
        apply_max_centered_threshold,
        invert_log2,
        log2_plus1,
        subtract_reference,
    )
    from pyinfercnv.smooth import smooth_pyramidinal  # noqa: PLC0415
    from pyinfercnv.center import center_cells  # noqa: PLC0415
    from pyinfercnv.cna import prune_outliers  # noqa: PLC0415

    X = np.asarray(hspike_counts, dtype=np.float64)
    ref_groups_for_subtract = {k: v for k, v in reference_indices.items()}

    # --- Step 4: log2(x+1) ---
    X = log2_plus1(X)

    # --- Step 8 (mirror): subtract reference (1st pass) ---
    X = subtract_reference(
        X,
        ref_groups=ref_groups_for_subtract,
        use_bounds=config.ref_subtract_use_mean_bounds,
    )

    # --- Step 9 (mirror): max-centered threshold ---
    X = apply_max_centered_threshold(X, threshold=config.max_centered_threshold)

    # --- Step 10 (mirror): smooth per fake chromosome ---
    unique_chrs = list(dict.fromkeys(gene_chr_labels))
    for cname in unique_chrs:
        col_mask = gene_chr_labels == cname
        col_idx = np.where(col_mask)[0]
        if col_idx.shape[0] < 2:
            continue
        X[:, col_idx] = smooth_pyramidinal(X[:, col_idx], window_length=config.window_length)

    # --- Step 11 (mirror): center cells (median) ---
    X = center_cells(X, method="median")

    # --- Step 12 (mirror): subtract reference (2nd pass) ---
    X = subtract_reference(
        X,
        ref_groups=ref_groups_for_subtract,
        use_bounds=config.ref_subtract_use_mean_bounds,
    )

    # --- Step 14 (mirror): invert_log2 — linear FC ---
    X = invert_log2(X)

    # R run() gates remove_outliers_norm, including its hspike mirror,
    # behind prune_outliers. Clipping only hspike would change calibration
    # while leaving the real expression matrix on a different scale.
    if config.prune_outliers:
        X = prune_outliers(
            X,
            method=config.outlier_method_bound,
            lower_bound=config.outlier_lower_bound,
            upper_bound=config.outlier_upper_bound,
        )

    return X.astype(np.float32), gene_chr_labels


def _gene_expr_mean_sd_by_cnv(
    hspike_linear: NDArray[np.float32],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract per-CNV-state ``(mu, sigma)`` from the Phase-1-replayed hspike.

    Mirror of R ``get_spike_dists`` + ``.get_gene_expr_by_cnv``
    (``inferCNV_HMM.R:15-68``). Values are in **linear FC space** (the
    post-step14 invert_log2, post-step16 outlier-prune hspike output),
    matching R's ``hspike_obj@expr.data`` at HMM time.
    Pools across observation cells (D.6) and across all fake chromosomes
    sharing the same CN level. Returns aligned (6,) arrays for
    :data:`I6_CNV_LEVELS_CALIBRATED`.
    """
    # Collect all observation cell rows
    obs_idx = np.concatenate([
        np.asarray(v, dtype=np.intp) for v in observation_indices.values()
    ])
    spike_expr = hspike_linear[obs_idx, :]  # (n_obs_cells, n_hspike_genes)

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
    hspike_linear: NDArray[np.float32],
    gene_chr_labels: NDArray[np.object_],
    observation_indices: dict[str, NDArray[np.intp]],
    *,
    num_rounds: int,
    max_num_cells: int,
    rng: np.random.Generator | RCompatibleRNG,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit ``lm(log(sigma(n)) ~ log(n))`` using the R aggregation convention.

    Mirror of R ``get_hspike_cnv_mean_sd_trend_by_num_cells_fit``
    (``inferCNV_HMM.R:154-212``).

    R idiosyncrasy (preserved verbatim):
        R's ``replicate(nrounds, sample(expr_vals, size=ncells, ...))``
        returns a matrix of shape ``(ncells, nrounds)``. R then calls
        ``rowMeans(vals)`` (``inferCNV_HMM.R:166``) — this averages across
        **rounds** *for each sample position*, not across positions within
        a round. The result is a length-``ncells`` vector of "mean across
        nrounds iid draws at position i". Its sd across positions is an
        empirical SE of the ``nrounds``-average estimator, which is
        ≈ ``sd(expr_vals) / sqrt(nrounds)`` and does NOT shrink with
        ``ncells``. This is what R's ``lm(log(sd) ~ log(num_cells))`` fits.

        Accordingly R's fitted slopes end up near 0 (not the -0.5 one
        would naïvely expect from a CLT argument). Reproducing this
        behaviour is required for R-parity i6 state_sigma values — see
        measured R output: ``cnv:0.01`` slope≈0.066, ``cnv:1`` slope≈0.041
        etc. Py previously computed the "correct" ``colMeans`` which gave
        -0.5 slopes and produced over-narrow sigmas, triggering i6
        false-positive state-0 calls.

    ncells=1 edge case (R): ``replicate`` returns a length-``nrounds``
    vector, falling into the ``else`` branch ``means <- mean(vals)`` (a
    single number), then ``sd(single number) = NA``. ``lm`` drops NA
    points. We emit NaN at ``n=1`` and mask before OLS.

    Returns
    -------
    slope, intercept
        Shape (6,) float64 each — per-state lm coefficients matching R.
    """
    obs_idx = np.concatenate([
        np.asarray(v, dtype=np.intp) for v in observation_indices.values()
    ])
    spike_expr = hspike_linear[obs_idx, :].astype(np.float64)

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

    # R builds its list in fake-chromosome order: neutral is sampled first.
    levels = list(cnv_to_vals) if isinstance(rng, RCompatibleRNG) else I6_CNV_LEVELS_CALIBRATED
    for level in levels:
        i = int(np.flatnonzero(I6_CNV_LEVELS_CALIBRATED == level)[0])
        expr_vals = cnv_to_vals.get(float(level), None)
        if expr_vals is None or expr_vals.shape[0] < 2:
            slopes[i] = 0.0
            intercepts[i] = 0.0
            continue

        sds = np.empty(max_num_cells, dtype=np.float64)
        for j, ncells in enumerate(n_cells_range):
            if ncells == 1:
                # R `replicate(nrounds, sample(..., size=1))` returns a
                # length-nrounds vector; `is(vals)` is not matrix; the
                # else branch computes `mean(vals)` (scalar), and
                # `sd(scalar) = NA`. Mirror that with NaN.
                if isinstance(rng, RCompatibleRNG):
                    rng.choice(expr_vals, size=num_rounds, replace=True)
                sds[j] = np.nan
                continue
            # Draw (ncells, num_rounds) — R `replicate` column-major
            samples = (rng.choice(expr_vals, size=(num_rounds, ncells), replace=True).T
                       if isinstance(rng, RCompatibleRNG)
                       else rng.choice(expr_vals, size=(ncells, num_rounds), replace=True))
            # R-parity "rowMeans" on the (ncells, nrounds) matrix:
            # mean across the nrounds columns → length-ncells vector.
            # This is R's inferCNV_HMM.R:166 behaviour (not a CLT-correct
            # colMeans across positions per round).
            means_per_position = samples.mean(axis=1)  # shape (ncells,)
            sds[j] = means_per_position.std(ddof=1) if ncells > 1 else np.nan

        # Mask NaN and zero-sd entries before log
        valid = np.isfinite(sds) & (sds > 0.0)
        if valid.sum() < 2:
            slopes[i] = 0.0
            intercepts[i] = 0.0
            continue
        log_sd = np.log(sds[valid])
        log_n_v = log_n[valid]
        X_mat = np.stack([log_n_v, np.ones_like(log_n_v)], axis=1)
        XtX = X_mat.T @ X_mat
        Xty = X_mat.T @ log_sd
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.array([0.0, 0.0])
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

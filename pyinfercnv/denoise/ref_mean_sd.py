"""Reference-mean±sd denoising — Phase 3 step 22 (Agent B3 implementation).

R source map
------------
* ``R/inferCNV_ops.R:2302-2346`` — ``clear_noise_via_ref_mean_sd``.
  Algorithm (L2304-2335):

    1. Pick reference cells; fall back to all observation cells if no
       reference is set (``inferCNV_ops.R:2308-2311``).
    2. ``mean_ref_vals = mean(vals)`` (scalar, global reference mean).
    3. ``mean_ref_sd  = mean(apply(vals, 2, sd, na.rm=TRUE)) * sd_amplifier``
       (mean of per-cell sd's across reference cells, scaled).
    4. ``upper = mean_ref_vals + mean_ref_sd``
       ``lower = mean_ref_vals - mean_ref_sd``
    5. ``expr[expr > lower & expr < upper] = mean_ref_vals``
       (in-place flattening; values on or outside the band are untouched).

* ``R/inferCNV_ops.R:2232-2264`` — ``clear_noise`` (called with explicit
  ``noise_filter`` threshold; band = ``[mean_ref - threshold,
  mean_ref + threshold]``; mean falls back to mean over the full matrix
  when no reference is set — *different* from the sd-path fallback).

Wired at ``inferCNV_ops.R:1560-1589`` (step 22):

    if denoise:
        if !is.na(noise_filter) && noise_filter > 0:
            clear_noise(threshold=noise_filter, ...)
        else:
            clear_noise_via_ref_mean_sd(sd_amplifier=sd_amplifier, ...)

Tier contract
-------------
**Tier 4 bit-exact** (``max_diff < 1e-10``). Pure mean / sd / boolean
clip — no RNG, no ties. The only Python-vs-R pitfall is
``np.std`` default ``ddof=0`` vs R ``sd`` ``ddof=1``; we use ``ddof=1``.

All arithmetic runs in float64 internally; the return array matches the
input dtype (float32 is preserved by casting the final scalar fill value
back down, but intermediate mean/sd are float64).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pyinfercnv.config import InferCNVConfig
    from pyinfercnv.result import InferCNVResult


def denoise_by_ref_mean_sd(
    cnv_matrix: np.ndarray,
    ref_indices: Sequence[int] | np.ndarray | None,
    *,
    noise_filter: float | None = None,
    sd_amplifier: float = 1.5,
    noise_logistic: bool = False,  # noqa: N803 — R arg name preserved (snake already)
) -> np.ndarray:
    """Flatten within-band values to the reference-mean centre.

    Pure-Python port of R ``clear_noise_via_ref_mean_sd`` (sd-path, the
    ``noise_filter=None`` default) and ``clear_noise`` (explicit-threshold
    path, ``noise_filter > 0``). Both share the "set values in a band
    around the reference mean to that mean" contract.

    Parameters
    ----------
    cnv_matrix
        Cells × genes/bins CNV matrix. Typical caller: Phase 2 output's
        :attr:`InferCNVResult.cnv_matrix_fc` (linear-FC space — this is
        what R's ``infercnv_obj@expr.data`` holds at step 22 once
        ``invert_log2`` has run). float32 or float64; the returned
        array matches the input dtype.
    ref_indices
        Integer indices into the cells (row) axis that flag reference
        cells. ``None`` or empty triggers R's fall-back:

        * sd-path (``noise_filter is None``): fall back to *all cells*
          (R uses ``unlist(observation_grouped_cell_indices)`` at
          ``inferCNV_ops.R:2309``; in a no-reference run every cell is an
          observation cell, so "all rows" is the correct mirror).
        * explicit-threshold path (``noise_filter > 0``): fall back to
          the mean of the *full matrix* (R ``inferCNV_ops.R:2246``).
    noise_filter
        Explicit ± bandwidth (same units as ``cnv_matrix``). When
        provided and ``> 0``, takes precedence over the ``sd_amplifier``
        path (R ``inferCNV_ops.R:1569-1581``). ``noise_filter == 0`` is
        a no-op in R (``inferCNV_ops.R:2236-2238``); we mirror that
        exactly — a straight copy is returned.
    sd_amplifier
        Multiplicative factor on the mean of per-cell sd's (R default
        1.5 at ``inferCNV_ops.R:303``). Ignored when ``noise_filter`` is
        provided.
    noise_logistic
        R ``inferCNV_ops.R:304,2326-2329`` — if True, apply a sigmoidal
        soft mask instead of a hard flatten. **Deferred to Phase 3a
        stretch** (plan doc §4); raises :class:`NotImplementedError`.

    Returns
    -------
    denoised
        Cells × genes/bins matrix with in-band entries flattened to the
        reference mean. Shape and dtype match ``cnv_matrix``.

    Raises
    ------
    NotImplementedError
        When ``noise_logistic=True`` (the sigmoidal path is not ported
        — see R ``depress_log_signal_midpt_val`` in
        ``inferCNV_heatmap.R:2783``).

    Notes
    -----
    * R uses strict ``>`` / ``<`` in the in-band mask
      (``inferCNV_ops.R:2275``, ``2335``); values exactly on the band
      boundary are **not** flattened. We mirror this exactly.
    * R's ``sd`` is ``ddof=1``; NumPy ``np.std`` defaults to ``ddof=0``.
      We pass ``ddof=1`` explicitly — required for bit-exact parity.
    * R's ``apply(vals, 2, sd)`` is per-cell sd because R orients
      ``expr.data`` as genes × cells (``MARGIN=2`` picks columns =
      cells). Python's ``cnv_matrix`` is cells × genes, so the per-cell
      axis is ``axis=1``.
    * ``mean`` is over the full reference sub-matrix (single scalar),
      not per-gene or per-cell.
    """
    if noise_logistic:
        raise NotImplementedError(
            "denoise_by_ref_mean_sd: noise_logistic=True path not yet "
            "ported (R `depress_log_signal_midpt_val` sigmoidal mask at "
            "inferCNV_heatmap.R:2783). Deferred — see plan doc §4."
        )

    if cnv_matrix.ndim != 2:
        raise ValueError(
            f"cnv_matrix must be 2-D (cells x bins); got shape {cnv_matrix.shape}"
        )

    # R line 2236-2238: `if (threshold == 0) return(infercnv_obj)`.
    if noise_filter is not None and float(noise_filter) == 0.0:
        return cnv_matrix.copy()

    # Normalise ref_indices; None / empty triggers the R fallback (L2308-2311
    # for the sd path, L2243-2246 for the explicit-threshold path — both
    # paths DO fall back, but to different subsets, hence separate branches
    # below).
    if ref_indices is None:
        ref_idx_arr: np.ndarray | None = None
    else:
        ref_idx_arr = np.asarray(ref_indices, dtype=np.intp).ravel()
        if ref_idx_arr.size == 0:
            ref_idx_arr = None

    orig_dtype = cnv_matrix.dtype
    # Promote to float64 for all mean/sd/compare arithmetic. This matches
    # R which keeps expr.data as double; float32 inputs would otherwise
    # drift away from R at the 1e-7 level and blow the 1e-10 bit-exact
    # floor.
    work = cnv_matrix.astype(np.float64, copy=True)

    if noise_filter is not None and float(noise_filter) > 0.0:
        # --- clear_noise (explicit-threshold) path ---
        # R ``inferCNV_ops.R:2240-2247``:
        #   if (has_reference_cells(...))
        #       mean_ref_vals = mean(expr.data[,ref_idx])
        #   else
        #       mean_ref_vals = mean(expr.data)    # FULL matrix, not obs-only
        if ref_idx_arr is None:
            mean_ref_vals = float(np.mean(work))
        else:
            vals = work[ref_idx_arr, :]
            mean_ref_vals = float(np.mean(vals))
        band = float(noise_filter)
    else:
        # --- clear_noise_via_ref_mean_sd (sd-path) default ---
        # R ``inferCNV_ops.R:2304-2316``:
        #   if (has_reference_cells) ref_idx = reference_grouped_cell_indices
        #   else                     ref_idx = observation_grouped_cell_indices
        #   vals = expr.data[, ref_idx]
        #   mean_ref_vals = mean(vals)
        #   mean_ref_sd   = mean(apply(vals, 2, sd, na.rm=TRUE)) * sd_amplifier
        if ref_idx_arr is None:
            # R falls back to obs cells; in a no-ref run that's every
            # cell. "All rows" matches R bit-exactly because the fallback
            # target set is defined as the full cell set.
            vals = work
        else:
            vals = work[ref_idx_arr, :]
        mean_ref_vals = float(np.mean(vals))
        # Per-cell sd. R's `apply(vals, 2, function(x) sd(x, na.rm=TRUE))`
        # is per-column = per-cell for R's genes×cells layout. Python's
        # cells×bins layout puts the per-cell axis at axis=1. ddof=1
        # mirrors R's `sd` (Bessel-corrected); NaN-safe via np.nanstd.
        per_cell_sd = np.nanstd(vals, axis=1, ddof=1)
        # `mean(apply(...))` is over the per-cell sd vector, plain mean
        # (no na.rm here — if a cell's sd came back NaN we let it
        # propagate like R would).
        mean_ref_sd = float(np.mean(per_cell_sd)) * float(sd_amplifier)
        band = mean_ref_sd

    lower = mean_ref_vals - band
    upper = mean_ref_vals + band
    # Strict inequalities — R ``inferCNV_ops.R:2275`` / ``2335``:
    #   expr.data[expr.data > lower_bound & expr.data < upper_bound] = mean_ref_vals
    mask = (work > lower) & (work < upper)
    work[mask] = mean_ref_vals

    # Cast back to the caller's dtype. For the bit-exact parity test the
    # caller will pass float64 so this is a no-op; for float32 callers the
    # cast reintroduces the 1e-7-level rounding that the float32 CNV
    # matrix already carries (no extra damage).
    if work.dtype != orig_dtype:
        return work.astype(orig_dtype, copy=False)
    return work


def _step22_denoise(
    result: "InferCNVResult",
    config: "InferCNVConfig",
) -> "InferCNVResult":
    """Phase 3 step 22 entry point — writes ``result.denoised_matrix``.

    Wraps :func:`denoise_by_ref_mean_sd` using the same input /
    orchestration contract the pipeline_phase3 orchestrator will adopt
    once B1/B2/B3 all land. This helper is intentionally decoupled from
    the orchestrator so Agent B3's module is independently testable.

    Input selection mirrors R's ``infercnv_obj@expr.data`` at step 22.
    When Bayes/mask are off (this agent's scope), the R object still
    holds the post-step16 **linear-FC** matrix (R invokes
    ``invert_log2`` at step 14; steps 17-20 mutate a separate
    ``hmm.infercnv_obj``, not the main ``infercnv_obj``). We therefore
    use :attr:`InferCNVResult.cnv_matrix_fc`. A future integration that
    chains mask_non_DE before denoise should swap in whatever masked
    matrix mask_de writes back.

    The reference-cell set is read from ``result.cell_meta['is_reference']``
    (Phase 1 contract; validated in :meth:`InferCNVResult.__post_init__`).

    Parameters
    ----------
    result
        Phase 2 (or post-mask Phase 3) result with populated
        :attr:`cnv_matrix_fc` and ``cell_meta``. Mutated: the
        :attr:`denoised_matrix` field is overwritten; no other fields
        change.
    config
        :class:`InferCNVConfig`. Consumed fields:
        ``denoise`` (toggle), ``noise_filter``, ``sd_amplifier``,
        ``noise_logistic``.

    Returns
    -------
    InferCNVResult
        The same ``result`` object (in-place mutation of
        ``denoised_matrix``).
    """
    # Short-circuit: denoise toggle off ⇒ leave result alone. The
    # orchestrator gates this already, but we double-guard so callers
    # that invoke ``_step22_denoise`` directly get a safe no-op.
    if not bool(config.denoise):
        return result

    if result.cnv_matrix_fc is None:
        raise ValueError(
            "_step22_denoise requires result.cnv_matrix_fc (post-step14 "
            "linear FC); got None. Run Phase 1 to completion first."
        )

    if "is_reference" not in result.cell_meta.columns:
        raise ValueError(
            "_step22_denoise requires result.cell_meta['is_reference']; "
            "column missing (should have been enforced by "
            "InferCNVResult.__post_init__)."
        )

    ref_mask = result.cell_meta["is_reference"].to_numpy()
    ref_idx = np.flatnonzero(ref_mask.astype(bool))
    # R's `has_reference_cells` returns FALSE when reference_grouped_cell_indices
    # is empty; the fallback paths differ between the sd and explicit-threshold
    # branches (see denoise_by_ref_mean_sd docstring). Passing None here lets
    # that helper pick the correct R-faithful fallback.
    ref_arg: np.ndarray | None = ref_idx if ref_idx.size > 0 else None

    denoised = denoise_by_ref_mean_sd(
        result.cnv_matrix_fc,
        ref_arg,
        noise_filter=config.noise_filter,
        sd_amplifier=config.sd_amplifier,
        noise_logistic=config.noise_logistic,
    )
    result.denoised_matrix = denoised
    return result


__all__ = ["denoise_by_ref_mean_sd", "_step22_denoise"]

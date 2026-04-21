"""Phase 2 pipeline — tumour subclustering + HMM state calls + CNV regions.

Composes (not replaces) :func:`pyinfercnv.pipeline.infercnv` output. The
canonical composition is:

    result_p1 = infercnv(adata, config=cfg, inplace=False, ...)
    result    = run_phase2(result_p1, config=cfg, adata=adata, ...)

or, equivalently, the top-level :func:`pyinfercnv.pipeline.infercnv` checks
``cfg.HMM`` and invokes this module internally. Phase 1's ``result.py``
schema is already frozen (G1 patch P1) for the fields written here:
``subclusters``, ``hmm_states``, ``hmm_states_i3``, ``cnv_regions``.

R-parity step mapping
---------------------
Continues from the Phase 1 mapping in :mod:`pyinfercnv.pipeline` (steps 1-14+16):

    15. define_signif_tumor_subclusters             -> :func:`_run_subclustering`
        (leiden / random_trees / qnorm backends already live in
         :mod:`pyinfercnv.subcluster`; we dispatch by
         ``cfg.tumor_subcluster_partition_method``)
    hspike. .build_and_add_hspike + get_spike_dists + sd-trend lm
                                                    -> :func:`_calibrate_hmm_emission`
                                                       (only when
                                                       ``cfg.HMM_type == "i6"``)
    17. predict_CNV_via_HMM_on_tumor_subclusters    -> :func:`_run_hmm_by_subcluster`
        (i6 path calls :func:`pyinfercnv.hmm.i6.predict_i6` on rowMeans per
         ``(subcluster, chromosome)`` with sd adjusted via
         :meth:`pyinfercnv.hmm.hspike.HspikeCalibration.sigmas_for_num_cells`;
         i3 path calls :func:`pyinfercnv.hmm.i3.estimate_i3_state_params`
         + :func:`pyinfercnv.hmm.i3.predict_i3` per subcluster.)
    17b. Convert state matrix to region BED        -> :func:`_build_cnv_regions`

Mapping to R entry: ``HMM=TRUE, BayesMaxPNormal=0`` (Bayes filter is Phase 3,
kept off here). ``HMM=FALSE`` short-circuits: ``run_phase2`` becomes a
pass-through returning ``result_phase1`` unchanged.

G1 patches in scope
-------------------
* **P3** — :func:`_calibrate_hmm_emission` re-exports
  :func:`pyinfercnv.hmm.hspike.calibrate_i6_emission`; ``__all__``
  discipline enforced downstream in ``hmm/__init__.py``.
* **P6 (deferred)** — current Phase 1 densifies once; Phase 2 works on the
  already-dense ``result.cnv_matrix``, so no additional densify here.
* **P11** — profile hooks on each block (subcluster, hspike, hmm, regions)
  write into the passed-in ``profile`` dict (which, when called from
  :func:`pyinfercnv.pipeline.infercnv`, is the same dict used for Phase 1
  blocks). Keys, continuing Phase 1's numeric prefix (G2 Q12):
  ``15_subcluster``, ``16_hspike_calibrate``, ``17_hmm``,
  ``18_cnv_regions``.

No new numba kernels
--------------------
Wave 2 adds no ``@njit`` beyond the existing
:mod:`pyinfercnv.kernels.hmm_viterbi_numba`. Subcluster-wise rowMeans,
RLE for region extraction, and the cross-Phase remap helper are all
NumPy/SciPy vectorized. (G2 Q4 documentation.)

Phase 1 G3 Q6 defensive
-----------------------
When ``reference_key``/``reference_cat`` match zero cells and
``cfg.HMM=True``, the i3 path would silently degrade (no reference → no
sigma estimate). This module raises :class:`ValueError` at the top of
:func:`run_phase2` when that situation is detected. The sibling defensive
check for Phase 1 (``result_phase1`` has zero ``is_reference=True`` rows
AND user passed ``reference_key`` explicitly) is tracked separately
(HANDOFF §5.7) and not implemented in this file.

Subcluster × chromosome HMM rationale
-------------------------------------
R ``predict_CNV_via_HMM_on_tumor_subclusters`` (inferCNV_HMM.R:345-408)
calls HMM on the rowMeans across cells of the subcluster, per chromosome,
with sd scaled by the lm fit from hspike. Phase 2 Wave 1 Agent-B's
``predict_i6`` / ``predict_i3`` operate per-cell, so we build an
intermediate aggregator in :func:`_run_hmm_by_subcluster`:

    for chr in chr_pos:
        for sc in unique(subclusters):
            cnv_slice = cnv_matrix[sc_members, chr_slice].mean(axis=0)
            sigmas_n  = calibration.sigmas_for_num_cells(len(sc_members))   # i6 only
            states    = predict_i6(cnv_slice[None, :], {chr: 0}, state_mus=mus, state_sigmas=sigmas_n, ...)
            for c in sc_members:
                hmm_states[c, chr_slice] = states[0]

This mirrors R's broadcasting of the subcluster's HMM trace back onto
each member cell (``hmm.data[chr_gene_idx,tumor_subcluster_cells_idx] <<- hmm_trace``
at R line 398). It also means all cells within a subcluster share an
identical state sequence per chromosome, consistent with R's output.

Matrix layout
-------------
* ``cnv_matrix`` input: ``(n_cells, n_bins)`` float32 C-order (from
  :attr:`InferCNVResult.cnv_matrix`, the log2-FC smoothed/centered matrix).
* ``hmm_states`` / ``hmm_states_i3`` output: ``(n_cells, n_bins)`` int8.
* ``subclusters`` output: ``(n_cells,)`` int32, ``-1`` for reference cells
  (R drops refs from subclustering; we signal the same via sentinel).
* ``cnv_regions`` output: long-format pandas DataFrame; one row per
  non-neutral state run, columns
  ``[cell_group, subcluster, chromosome, bin_start, bin_end, state, cn]``.

Skeleton status
---------------
G2-pending API skeleton — all function bodies raise
``NotImplementedError("skeleton — G2 pending")``. Implementation lands in
Wave 2 (after codex G2 adjudication on API + subcluster aggregation logic
+ profile hook placement).
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

    from pyinfercnv.config import InferCNVConfig
    from pyinfercnv.hmm.hspike import HspikeCalibration
    from pyinfercnv.result import InferCNVResult


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

#: i6 neutral state index (CN=1.0); used as sentinel in region extraction.
_I6_NEUTRAL_IDX: int = 2

#: i3 neutral state index.
_I3_NEUTRAL_IDX: int = 1

#: Sentinel for reference cells in the subclusters array (R drops refs from
#: subclustering; we keep the slot but mark with -1).
_SUBCLUSTER_REF_SENTINEL: np.int32 = np.int32(-1)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def run_phase2(
    result_phase1: "InferCNVResult",
    adata: "AnnData",
    *,
    config: "InferCNVConfig",
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    random_state: int = 0,
    profile: dict[str, Any] | None = None,
) -> "InferCNVResult":
    """Orchestrate R steps 15 (subcluster), hspike (i6 calibration), 17 (HMM).

    Short-circuits when ``config.HMM`` is False: returns ``result_phase1``
    unchanged (the subclustering step is gated by HMM in this port; use
    standalone :mod:`pyinfercnv.subcluster` backends for subcluster-only
    needs). This differs from R where subclustering also runs without HMM,
    but Phase 2 scope aligns with ``HMM=TRUE`` end-to-end per the master
    spec §2.2.

    Parameters
    ----------
    result_phase1
        Output of Phase 1 :func:`pyinfercnv.pipeline.infercnv` with
        ``inplace=False``. Must have ``cnv_matrix`` (log2-FC), ``chr_pos``,
        ``cell_meta`` with ``is_reference``, and ``ref_counts_raw``
        (threaded by G1 P2 for hspike calibration).
    adata
        Original AnnData — needed only to resolve ``reference_key``/
        ``reference_cat`` to cell indices when ``config.HMM_type == "i3"``
        or when Phase 2 needs to verify reference cells match the Phase 1
        ``is_reference`` column.
    config
        :class:`InferCNVConfig`. Fields honoured:
        ``HMM``, ``HMM_type``, ``HMM_transition_prob``, ``HMM_i3_pval``,
        ``tumor_subcluster_partition_method`` (future field in Phase 2;
        default "leiden" via helper), ``analysis_mode``, ``cutoff``,
        ``window_length`` (for hspike replay), plus the preprocess knobs.
    reference_key, reference_cat
        Required when ``config.HMM_type == "i6"``: they drive the global→
        ref-local index remap via :func:`_remap_ref_groups_to_local` (G2
        Q10). When ``config.HMM_type == "i3"``, also used by
        :func:`_validate_reference_and_raise` to enforce Phase 1 G3 Q6
        defensive. When ``None`` for the i6 path and the reference pool is
        non-empty, the module falls back to a single-group
        ``'normalsToUse'`` bundle over all reference cells (R's
        reference-less branch in ``inferCNV_hidden_spike.R:19-26``).
    random_state
        Propagated to hspike simulation and any stochastic subcluster
        backend (leiden / random_trees; qnorm is deterministic).
    profile
        Optional dict mutated in place (G1 P11). When called from
        :func:`pyinfercnv.pipeline.infercnv`, this is the same dict as
        Phase 1 so block names compose into a single timeline.

    Returns
    -------
    InferCNVResult
        A *new* :class:`InferCNVResult` (not mutation of ``result_phase1``)
        with populated ``subclusters``, ``hmm_states`` or
        ``hmm_states_i3``, and ``cnv_regions``. All Phase 1 fields are
        passed through unchanged.

    Raises
    ------
    ValueError
        * If ``result_phase1.cnv_matrix`` lacks the dtype/layout contract.
        * If ``config.HMM_type == "i6"`` and ``result_phase1.ref_counts_raw is None``
          (no reference cells → hspike impossible).
        * If ``reference_key`` was supplied but no cells matched
          ``reference_cat`` (Phase 1 G3 Q6 defensive, applied to Phase 2's
          own i3 re-estimation path).
    """
    raise NotImplementedError("skeleton — G2 pending")


# --------------------------------------------------------------------------- #
# Private orchestration helpers                                               #
# --------------------------------------------------------------------------- #


def _run_subclustering(
    cnv_matrix: NDArray[np.float32],
    is_reference: NDArray[np.bool_],
    *,
    config: "InferCNVConfig",
    random_state: int,
    profile: dict[str, Any] | None,
) -> NDArray[np.int32]:
    """Run the configured subcluster backend on the non-reference cells.

    Dispatches to one of:
        * :func:`pyinfercnv.subcluster.leiden_subcluster`
        * :func:`pyinfercnv.subcluster.random_tree_subcluster`
        * :func:`pyinfercnv.subcluster.qnorm_subcluster`
    based on ``config.tumor_subcluster_partition_method`` (Phase 2 config
    extension; see NB below).

    Reference cells receive :data:`_SUBCLUSTER_REF_SENTINEL`. R drops them
    from subclustering but we preserve the vector length to match the
    Phase 1 cell axis.

    Returns
    -------
    labels
        Shape (n_cells,) int32. Non-reference cells have cluster ids
        0..K-1; reference cells have -1.

    Notes
    -----
    ``InferCNVConfig`` currently lacks a ``tumor_subcluster_partition_method``
    field (Phase 2 plan Task 49 adds it). During G2 review, codex may
    flag this as an implicit dependency; the implementation will add the
    field to config.py in parallel. Until then the helper defaults to
    ``'leiden'`` when absent.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _calibrate_hmm_emission(
    ref_counts_raw: NDArray[np.float32],
    ref_groups_local: Mapping[str, NDArray[np.intp]] | None,
    *,
    config: "InferCNVConfig",
    random_state: int,
    profile: dict[str, Any] | None,
) -> "HspikeCalibration":
    """Thin wrapper over :func:`pyinfercnv.hmm.hspike.calibrate_i6_emission`.

    Exists to centralise the psutil hook placement (G1 P11 key
    ``p2_hspike_calibrate``) and to supply Phase-2-level defaults for
    simulation kwargs without polluting the public hspike signature.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _run_hmm_by_subcluster(
    cnv_matrix: NDArray[np.float32],
    chr_pos: dict[str, int],
    subclusters: NDArray[np.int32],
    is_reference: NDArray[np.bool_],
    *,
    hmm_type: str,
    transition_prob: float,
    i6_calibration: "HspikeCalibration | None",
    i3_mus: NDArray[np.float64] | None,
    i3_sigmas: NDArray[np.float64] | None,
) -> NDArray[np.int8]:
    """Predict CNV states by aggregating across subclusters per chromosome.

    Mirror of R ``predict_CNV_via_HMM_on_tumor_subclusters``
    (``inferCNV_HMM.R:345-408``). Algorithm:

        1. Determine neutral state idx and ``N_STATES`` from ``hmm_type``.
        2. Initialise output ``hmm_states = full(shape, neutral_idx, dtype=int8)``.
        3. For each subcluster ``s`` (skipping refs marked with sentinel):
            a. For each chromosome block ``[start, end)`` in ``chr_pos``:
                * ``x = cnv_matrix[s_members, start:end].mean(axis=0)``
                * Build the per-call state_mus / state_sigmas arrays
                  (G2 Q8 — Wave-1 ``predict_i6`` / ``predict_i3`` expect
                  plain ndarrays, not the :class:`HspikeCalibration`
                  object, so this helper unpacks them):

                  - i6:  ``state_mus    = i6_calibration.state_mus``           # shape (6,)
                         ``state_sigmas = i6_calibration.sigmas_for_num_cells(len(s_members))``
                  - i3:  ``state_mus    = i3_mus``                              # shape (3,)
                         ``state_sigmas = i3_sigmas``                           # shape (3,)

                * Call the matching
                  ``predict_i6(x[None, :], {chr: 0}, transition_prob=t,
                  state_mus=state_mus, state_sigmas=state_sigmas)`` or the
                  equivalent ``predict_i3`` overload.
                * Broadcast the returned (1, len) trace to all ``s_members``.
        4. Reference cells retain ``neutral_idx`` everywhere (R behaviour:
           refs are not HMM-called; displays as neutral in downstream).

    ``chr_pos`` column ordering is preserved — callers must not reorder.

    Raises
    ------
    ValueError
        ``hmm_type in {"i6", "i3"}`` violated, or shape mismatches.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _build_cnv_regions(
    hmm_states: NDArray[np.int8],
    subclusters: NDArray[np.int32],
    chr_pos: dict[str, int],
    *,
    hmm_type: str,
    cell_group_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run-length-encode non-neutral HMM state segments into a long DataFrame.

    One row per run-length within a ``(subcluster, chromosome)`` cell; runs
    of the neutral state are skipped. Output columns (R maps loosely to
    ``HMM_CNV_predictions.*.cell_groupings`` + ``.bed`` files):

        ``cell_group``    — reference group name (for normal calls) or
                            ``f"subcluster_{sc}"``. ``None`` for refs.
        ``subcluster``    — int32. ``-1`` for reference cells.
        ``chromosome``    — str (real chr name from Phase 1 ``chr_pos``).
        ``bin_start``     — inclusive column index.
        ``bin_end``       — inclusive column index (R is inclusive; mirrors
                            ``HMM.txt`` BED output).
        ``state``         — int8 HMM state index.
        ``cn``            — float64 CN ratio for that state (via
                            :data:`pyinfercnv.hmm.hspike.I6_CNV_LEVELS_CALIBRATED`
                            for i6 or ``{-1, 0, +1}``-to-CN mapping for i3).

    Mirrors R per-subcluster BED output: the Phase 3 denoise step and any
    viz code should consume this DataFrame rather than re-scanning
    ``hmm_states``.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _validate_reference_and_raise(
    adata: "AnnData",
    reference_key: str | None,
    reference_cat: str | Sequence[str] | None,
    is_reference_fallback: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Guard against zero-match reference (Phase 1 G3 Q6, applied to Phase 2).

    If the caller supplied ``reference_key`` + ``reference_cat`` explicitly,
    verify at least one cell matches. Otherwise fall back to
    ``is_reference_fallback`` (the Phase 1 ``cell_meta.is_reference``
    column). Returns a bool mask over ``adata.n_obs``.

    Raises
    ------
    ValueError
        When explicit ``reference_key``/``reference_cat`` match zero cells.
        The error message lists the observed categories so the caller can
        correct the typo.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _remap_ref_groups_to_local(
    adata: "AnnData",
    is_reference: NDArray[np.bool_],
    reference_key: str | None,
    reference_cat: str | Sequence[str] | None,
) -> dict[str, NDArray[np.intp]] | None:
    """Convert global reference group membership into row-of-``ref_counts_raw`` indices.

    G2 Q10 — :attr:`InferCNVResult.ref_counts_raw` contains only the rows of
    cells flagged ``is_reference=True`` (Phase 1 pipeline at
    ``pipeline.py:151-155``), in the original ``adata.obs`` order restricted
    to that mask. Meanwhile :class:`pyinfercnv.hmm.hspike.HspikeCalibration`
    (via :func:`pyinfercnv.hmm.hspike.calibrate_i6_emission`) expects
    ``ref_groups_local`` keyed by group name with values indexing
    **rows of ref_counts_raw** (0-based, local), not global
    ``adata.obs`` indices.

    Returns
    -------
    dict
        Keys: each entry in ``reference_cat`` (str form). Values: integer
        arrays indexing into ``ref_counts_raw`` rows. Returns ``None`` when
        ``reference_key`` is ``None`` (caller should fall back to
        ``{'normalsToUse': arange(n_ref)}`` as per R's reference-less
        branch at ``inferCNV_hidden_spike.R:19-26``).

    Notes
    -----
    Implementation sketch (Wave 2):

        ref_positions = np.where(is_reference)[0]            # global idx of rows kept
        local_map = {g: i for i, g in enumerate(ref_positions)}
        for cat in reference_cat:
            global_cells = np.where(adata.obs[reference_key] == cat)[0]
            result[str(cat)] = np.array([local_map[g] for g in global_cells if g in local_map], dtype=np.intp)

    The implementation must survive the case where a category in
    ``reference_cat`` has zero members (empty array kept for downstream
    group-iteration compatibility) but the full set returning empty is
    caught by :func:`_validate_reference_and_raise` upstream.
    """
    raise NotImplementedError("skeleton — G2 pending")


def _profile_block(
    profile: dict[str, Any] | None,
    name: str,
    t0: float,
    rss_before: float | None,
) -> None:
    """Thin wrapper mirroring :func:`pyinfercnv.pipeline._profile_block` for
    consistent profile-dict layout. Imported at call site to avoid a circular
    module dependency between pipeline.py and pipeline_phase2.py.
    """
    raise NotImplementedError("skeleton — G2 pending")


__all__ = [
    "run_phase2",
]

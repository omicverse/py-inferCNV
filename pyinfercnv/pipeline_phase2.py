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
* ``subclusters`` output: ``(n_cells,)`` int32. When
  ``config.cluster_by_groups=True`` (default, R parity) every cell has a
  non-negative subcluster id; reference cells receive their own group's
  subcluster ids just like tumor cells. When
  ``config.cluster_by_groups=False`` the legacy behaviour kicks in:
  non-ref cells get 0..K-1 and reference cells receive the ``-1``
  sentinel.
* ``cnv_regions`` output: long-format pandas DataFrame; one row per
  non-neutral state run, columns
  ``[cell_group, subcluster, chromosome, bin_start, bin_end, state, cn]``.
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

#: i3 CN delta values for states (DEL, neutral, AMP) -> signed delta
_I3_CN_DELTA: dict[int, float] = {0: -1.0, 1: 0.0, 2: 1.0}


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
        ``tumor_subcluster_partition_method`` (default "leiden"),
        ``tumor_subcluster_n_seeds`` (``rbest`` if >1; default 1),
        ``tumor_subcluster_min_size`` (small-cluster merge; default None),
        ``analysis_mode`` (must be "subclusters" in Phase 2; "samples"
        and "cells" are Phase 3), ``cutoff``, ``window_length`` (for hspike replay),
        plus the preprocess knobs.
    reference_key, reference_cat
        **Optional in all modes.** When supplied and both
        ``cluster_by_groups=True`` and ``HMM_type=="i6"``, the pair drives
        the global -> ref-local index remap via
        :func:`_remap_ref_groups_to_local` (G2 Q10), so each annotation
        category maps to its own hspike ref-group bundle. When ``None``
        and the Phase-1 ``is_reference`` mask is non-empty, all reference
        cells collapse into a single-group ``'normalsToUse'`` bundle over
        the full reference pool (matches R's reference-less branch in
        ``inferCNV_hidden_spike.R:19-26``). Either mode is supported end-
        to-end for i6; the per-group path is preferred because it
        matches ``infercnv::run(cluster_by_groups=TRUE, ...)`` parity.
        For ``HMM_type=="i3"`` these kwargs are used only by
        :func:`_validate_reference_and_raise` to enforce the Phase 1
        G3 Q6 defensive ("supplied key with zero matches is a user bug,
        not a silent fallback"); the i3 state parameter estimator works
        from ``is_reference`` alone.
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
          (no reference cells -> hspike impossible).
        * If ``reference_key`` was supplied but no cells matched
          ``reference_cat`` (Phase 1 G3 Q6 defensive, applied to Phase 2's
          own i3 re-estimation path).
    """
    from pyinfercnv.result import InferCNVResult

    # Short-circuit when HMM is disabled
    if not config.HMM:
        return result_phase1

    # Use profile dict from result_phase1 if none provided (shared timeline)
    if profile is None:
        profile = result_phase1.profile if result_phase1.profile is not None else {}

    # Derive reference mask from Phase 1 cell_meta
    is_reference: NDArray[np.bool_] = result_phase1.cell_meta["is_reference"].to_numpy().astype(bool)

    # Validate reference and get canonical mask
    is_reference = _validate_reference_and_raise(
        adata, reference_key, reference_cat, is_reference
    )

    cnv_matrix = result_phase1.cnv_matrix
    chr_pos = result_phase1.chr_pos

    # Step 15 — subclustering. When reference_key is available we honour
    # R's cluster_by_groups=TRUE: iterate over unique category labels in
    # adata.obs[reference_key] and run leiden per group.
    group_labels = None
    if reference_key is not None and reference_key in adata.obs.columns:
        group_labels = adata.obs[reference_key].to_numpy()

    subclusters = _run_subclustering(
        cnv_matrix, is_reference,
        config=config, random_state=random_state, profile=profile,
        group_labels=group_labels,
    )

    # HMM branch
    hmm_type = config.HMM_type
    hmm_states_i6: NDArray[np.int8] | None = None
    hmm_states_i3: NDArray[np.int8] | None = None

    if hmm_type == "i6":
        if result_phase1.ref_counts_raw is None:
            raise ValueError(
                "config.HMM_type='i6' requires ref_counts_raw (hspike calibration), "
                "but result_phase1.ref_counts_raw is None. "
                "Ensure reference cells are present in the Phase 1 run."
            )
        # G2 Q10 — remap global ref indices to local ref_counts_raw row indices
        ref_groups_local = _remap_ref_groups_to_local(
            adata, is_reference, reference_key, reference_cat
        )
        # Fallback: single group over all ref rows
        if ref_groups_local is None:
            n_ref = int(is_reference.sum())
            ref_groups_local = {"normalsToUse": np.arange(n_ref, dtype=np.intp)}

        _t_hspike = time.perf_counter()
        _rss_hspike = _rss_mb()
        cal = _calibrate_hmm_emission(
            result_phase1.ref_counts_raw, ref_groups_local,
            config=config, random_state=random_state, profile=profile,
        )
        _profile_block(profile, "16_hspike_calibrate", _t_hspike, _rss_hspike)
        hmm_states_i6 = _run_hmm_by_subcluster(
            cnv_matrix, chr_pos, subclusters, is_reference,
            hmm_type="i6",
            transition_prob=config.HMM_transition_prob,
            i6_calibration=cal,
            i3_mus=None,
            i3_sigmas=None,
            profile=profile,
        )

    elif hmm_type == "i3":
        import pyinfercnv.hmm.i3 as _i3
        ref_idx = np.where(is_reference)[0]
        mus, sigs = _i3.estimate_i3_state_params(
            cnv_matrix, ref_idx, config.HMM_i3_pval
        )
        hmm_states_i3 = _run_hmm_by_subcluster(
            cnv_matrix, chr_pos, subclusters, is_reference,
            hmm_type="i3",
            transition_prob=config.HMM_transition_prob,
            i6_calibration=None,
            i3_mus=mus,
            i3_sigmas=sigs,
            profile=profile,
        )
    else:
        raise ValueError(f"config.HMM_type must be 'i6' or 'i3', got {hmm_type!r}")

    # Step 17b — build CNV regions
    active_states = hmm_states_i6 if hmm_type == "i6" else hmm_states_i3
    t0 = time.perf_counter()
    rss0 = _rss_mb()
    cnv_regions = _build_cnv_regions(
        active_states, subclusters, chr_pos,
        hmm_type=hmm_type,
    )
    _profile_block(profile, "18_cnv_regions", t0, rss0)

    # Build new result — reuse Phase 1 arrays BY REFERENCE (G2 Q5)
    result = InferCNVResult(
        chr_pos=result_phase1.chr_pos,
        cnv_matrix=result_phase1.cnv_matrix,
        cnv_matrix_fc=result_phase1.cnv_matrix_fc,
        cell_meta=result_phase1.cell_meta,
        gene_values=result_phase1.gene_values,
        ref_counts_raw=result_phase1.ref_counts_raw,
        subclusters=subclusters,
        hmm_states=hmm_states_i6,
        hmm_states_i3=hmm_states_i3,
        cnv_regions=cnv_regions,
        profile=profile,
    )
    return result


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
    group_labels: NDArray[np.object_] | None = None,
) -> NDArray[np.int32]:
    """Run the configured subcluster backend, per-observation-group when
    ``config.cluster_by_groups`` is True and ``group_labels`` is supplied.

    Dispatches to one of:
        * :func:`pyinfercnv.subcluster.leiden_subcluster`
        * :func:`pyinfercnv.subcluster.random_tree_subcluster`
        * :func:`pyinfercnv.subcluster.qnorm_subcluster`
    based on ``config.tumor_subcluster_partition_method``.

    When ``group_labels is not None`` and ``config.cluster_by_groups``,
    this function mirrors R ``define_signif_tumor_subclusters(
    cluster_by_groups=TRUE)`` exactly: it iterates over the unique
    categories in ``group_labels`` (which normally comes from the
    ``reference_key`` column of ``adata.obs``), runs the partition
    backend independently within each group, and assigns globally
    unique subcluster ids. Reference cells also receive real subcluster
    ids (no sentinel) so the HMM step processes them, matching R.

    When ``group_labels is None`` or ``cluster_by_groups=False``,
    falls back to the legacy behaviour: a single backend call on all
    non-reference cells with reference cells marked by
    :data:`_SUBCLUSTER_REF_SENTINEL`.

    Returns
    -------
    labels
        Shape (n_cells,) int32. Non-negative ids 0..K-1 (per-group
        mode) or 0..K-1 on non-refs + -1 sentinel on refs (fallback).
    """
    n_cells = cnv_matrix.shape[0]
    non_ref_mask = ~is_reference
    n_non_ref = int(non_ref_mask.sum())

    if n_non_ref == 0:
        raise ValueError(
            "No non-reference cells found for subclustering. "
            "All cells are marked as reference (is_reference=True). "
            "Phase 2 requires at least one tumour cell."
        )

    t0 = time.perf_counter()
    rss0 = _rss_mb()

    method = getattr(config, "tumor_subcluster_partition_method", "leiden")
    cluster_by_groups = bool(getattr(config, "cluster_by_groups", True))

    def _partition(X: NDArray[np.float32]) -> NDArray[np.int32]:
        if method == "leiden":
            from pyinfercnv.subcluster.leiden import leiden_subcluster
            return leiden_subcluster(
                X,
                random_state=random_state,
                n_seeds=int(getattr(config, "tumor_subcluster_n_seeds", 1)),
                min_subcluster_size=getattr(
                    config, "tumor_subcluster_min_size", None
                ),
            )
        if method == "random_trees":
            from pyinfercnv.subcluster.random_trees import random_tree_subcluster
            return random_tree_subcluster(X, random_state=random_state)
        if method == "qnorm":
            from pyinfercnv.subcluster.qnorm import qnorm_subcluster
            return qnorm_subcluster(X)
        raise ValueError(
            f"tumor_subcluster_partition_method must be one of "
            f"('leiden', 'random_trees', 'qnorm'), got {method!r}"
        )

    if group_labels is not None and cluster_by_groups:
        # Per-group subclustering (R cluster_by_groups=TRUE).
        group_labels = np.asarray(group_labels)
        if group_labels.shape[0] != n_cells:
            raise ValueError(
                f"group_labels length {group_labels.shape[0]} != n_cells {n_cells}"
            )

        labels = np.full(n_cells, -1, dtype=np.int32)
        # Stable group ordering: first-occurrence order in group_labels.
        seen: dict[object, None] = {}
        for g in group_labels.tolist():
            if g not in seen:
                seen[g] = None
        group_order = list(seen.keys())

        next_id = 0
        for g in group_order:
            member_mask = group_labels == g
            member_idx = np.where(member_mask)[0]
            if member_idx.size == 0:
                continue
            sub_X = cnv_matrix[member_idx]
            local = np.asarray(_partition(sub_X), dtype=np.int32)
            # Offset so ids are globally unique across groups.
            # Each distinct local id gets a fresh global id.
            _uniq = sorted({int(v) for v in local})
            remap = {v: next_id + i for i, v in enumerate(_uniq)}
            global_labels = np.array([remap[int(v)] for v in local],
                                      dtype=np.int32)
            labels[member_idx] = global_labels
            next_id += len(_uniq)
    else:
        # Fallback (legacy / when no group_labels supplied): pooled
        # non-ref leiden + sentinel for refs. Preserved for callers
        # that do not pass reference_key.
        non_ref_matrix = cnv_matrix[non_ref_mask]
        non_ref_labels = np.asarray(_partition(non_ref_matrix), dtype=np.int32)
        labels = np.full(n_cells, _SUBCLUSTER_REF_SENTINEL, dtype=np.int32)
        non_ref_idx = np.where(non_ref_mask)[0]
        labels[non_ref_idx] = non_ref_labels

    _profile_block(profile, "15_subcluster", t0, rss0)
    return labels


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
    ``16_hspike_calibrate``) and to supply Phase-2-level defaults for
    simulation kwargs without polluting the public hspike signature.
    """
    from pyinfercnv.hmm.hspike import calibrate_i6_emission  # late import — allows monkeypatching

    cal = calibrate_i6_emission(
        ref_counts_raw,
        ref_groups_local,
        config=config,
        sim_method="meanvar",
        aggregate_normals=False,
        random_state=random_state,
        profile=profile,
    )
    return cal


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
    profile: dict[str, Any] | None = None,
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
    if hmm_type not in ("i6", "i3"):
        raise ValueError(f"hmm_type must be 'i6' or 'i3', got {hmm_type!r}")

    n_cells, n_bins = cnv_matrix.shape
    neutral_idx = _I6_NEUTRAL_IDX if hmm_type == "i6" else _I3_NEUTRAL_IDX

    hmm_states = np.full((n_cells, n_bins), neutral_idx, dtype=np.int8)

    # Build chromosome start/end pairs
    chroms = list(chr_pos.keys())
    starts = [chr_pos[c] for c in chroms]
    ends = starts[1:] + [n_bins]

    unique_subclusters = np.unique(subclusters)
    # Skip the reference sentinel
    tumor_subclusters = unique_subclusters[unique_subclusters != _SUBCLUSTER_REF_SENTINEL]

    t0 = time.perf_counter()
    rss0 = _rss_mb()

    if hmm_type == "i6":
        from pyinfercnv.hmm.i6 import predict_i6

        for sc in tumor_subclusters:
            members = np.where(subclusters == sc)[0]
            n_members = len(members)
            sigmas = i6_calibration.sigmas_for_num_cells(n_members)  # shape (6,)
            state_mus = i6_calibration.state_mus

            for chrom, start, end in zip(chroms, starts, ends):
                if end <= start:
                    continue
                x = cnv_matrix[members, start:end].mean(axis=0)  # (n_bins_chr,)
                # predict_i6 expects chr_pos with first start == 0
                states = predict_i6(
                    x[np.newaxis, :],
                    {chrom: 0},
                    transition_prob=transition_prob,
                    state_mus=state_mus,
                    state_sigmas=sigmas,
                )  # shape (1, n_bins_chr)
                hmm_states[np.ix_(members, np.arange(start, end))] = states[0]

    else:  # i3
        from pyinfercnv.hmm.i3 import predict_i3

        for sc in tumor_subclusters:
            members = np.where(subclusters == sc)[0]

            for chrom, start, end in zip(chroms, starts, ends):
                if end <= start:
                    continue
                x = cnv_matrix[members, start:end].mean(axis=0)  # (n_bins_chr,)
                states = predict_i3(
                    x[np.newaxis, :],
                    {chrom: 0},
                    transition_prob=transition_prob,
                    state_mus=i3_mus,
                    state_sigmas=i3_sigmas,
                )  # shape (1, n_bins_chr)
                hmm_states[np.ix_(members, np.arange(start, end))] = states[0]

    _profile_block(profile, "17_hmm", t0, rss0)

    return hmm_states


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
    neutral_idx = _I6_NEUTRAL_IDX if hmm_type == "i6" else _I3_NEUTRAL_IDX

    if hmm_type == "i6":
        from pyinfercnv.hmm.i6 import I6_CNV_LEVELS
        cn_map: dict[int, float] = {i: float(I6_CNV_LEVELS[i]) for i in range(len(I6_CNV_LEVELS))}
    else:
        cn_map = {0: -1.0, 1: 0.0, 2: 1.0}

    chroms = list(chr_pos.keys())
    n_bins = hmm_states.shape[1]
    starts = [chr_pos[c] for c in chroms]
    ends = starts[1:] + [n_bins]

    rows: list[dict] = []

    unique_subclusters = np.unique(subclusters)
    tumor_subclusters = unique_subclusters[unique_subclusters != _SUBCLUSTER_REF_SENTINEL]

    for sc in tumor_subclusters:
        members = np.where(subclusters == sc)[0]
        cell_group = f"subcluster_{sc}"

        # All cells in same subcluster share the same HMM trace — pick first member
        # (by construction from _run_hmm_by_subcluster)
        rep = members[0]

        for chrom, start, end in zip(chroms, starts, ends):
            if end <= start:
                continue
            trace = hmm_states[rep, start:end]

            # RLE over this chromosome trace
            i = 0
            while i < len(trace):
                state = int(trace[i])
                j = i + 1
                while j < len(trace) and int(trace[j]) == state:
                    j += 1
                # Run of state from i to j-1 (inclusive), in global bin coords
                bin_start = start + i
                bin_end = start + j - 1  # inclusive
                if state != neutral_idx:
                    rows.append({
                        "cell_group": cell_group,
                        "subcluster": np.int32(sc),
                        "chromosome": chrom,
                        "bin_start": int(bin_start),
                        "bin_end": int(bin_end),
                        "state": np.int8(state),
                        "cn": cn_map.get(state, float("nan")),
                    })
                i = j

    if not rows:
        return pd.DataFrame(
            columns=["cell_group", "subcluster", "chromosome", "bin_start", "bin_end", "state", "cn"]
        )

    df = pd.DataFrame(rows)
    df["subcluster"] = df["subcluster"].astype(np.int32)
    df["state"] = df["state"].astype(np.int8)
    df["cn"] = df["cn"].astype(np.float64)
    return df


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
    if reference_key is not None and reference_cat is not None:
        cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
        if reference_key not in adata.obs.columns:
            observed = list(adata.obs.columns)
            raise ValueError(
                f"reference_key={reference_key!r} not found in adata.obs. "
                f"Available columns: {observed}"
            )
        obs_series = adata.obs[reference_key]
        mask = obs_series.isin(cats).to_numpy().astype(bool)
        if mask.sum() == 0:
            observed_cats = sorted(obs_series.unique().tolist())
            raise ValueError(
                f"reference_cat={cats!r} matched zero cells in "
                f"adata.obs[{reference_key!r}]. "
                f"Observed categories: {observed_cats}"
            )
        return mask
    else:
        return is_reference_fallback.astype(bool)


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
    if reference_key is None:
        return None

    cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
    if cats is None or len(cats) == 0:
        return None

    # Map global row index -> local ref_counts_raw row index
    ref_positions = np.where(is_reference)[0]  # global indices of ref cells, in order
    local_map: dict[int, int] = {int(g): i for i, g in enumerate(ref_positions)}

    result: dict[str, NDArray[np.intp]] = {}
    for cat in cats:
        global_cells = np.where(adata.obs[reference_key].to_numpy() == cat)[0]
        local_indices = np.array(
            [local_map[int(g)] for g in global_cells if int(g) in local_map],
            dtype=np.intp,
        )
        result[str(cat)] = local_indices

    return result


def _rss_mb() -> float | None:
    """Return current process RSS in MB via psutil, or None if unavailable."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


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
    if profile is None:
        return
    import logging
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    profile[name] = {
        "wallclock_s": float(elapsed),
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_delta_mb": (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None,
    }
    _log = logging.getLogger("pyinfercnv.profile")
    _log.info("block=%s wallclock=%.3fs rss_after=%s", name, elapsed, rss_after)


__all__ = [
    "run_phase2",
]

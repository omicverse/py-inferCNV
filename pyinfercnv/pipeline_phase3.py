"""Phase 3 top-level orchestrator — BayesNet + mask_non_DE + denoise.

Mirrors the shape of :func:`pyinfercnv.pipeline_phase2.run_phase2` but
runs after HMM state calls. Composes (not replaces) Phase 2 output.
Canonical composition::

    result_p1 = infercnv(adata, config=cfg, inplace=False, ...)
    result_p2 = run_phase2(result_p1, adata, config=cfg, ...)
    result    = run_phase3(result_p2, config=cfg, ...)

Top-level :func:`pyinfercnv.pipeline.infercnv` invokes ``run_phase3``
automatically when ``cfg.HMM=True`` (R-faithful step 20 per
``inferCNV_ops.R:1463-1499``) or any of ``BayesMaxPNormal > 0``,
``mask_nonDE_genes``, or ``denoise`` is set.

R-parity step mapping (inferCNV_ops.R step_count comments inline)
-----------------------------------------------------------------
* **step 18** — ``inferCNVBayesNet`` (``inferCNV_ops.R:1362-1393``).
  Guard: ``HMM == TRUE && BayesMaxPNormal > 0``.
  Py: :func:`pyinfercnv.bayesnet.gibbs.run_bayesnet_gibbs`.
* **step 19** — ``filterHighPNormals`` (``inferCNV_ops.R:1405-1454``).
  Consumes step-18 posterior; rewrites
  ``hmm.infercnv_obj@expr.data``. Same guard as step 18.
  Py: posterior-filter helper collocated with BayesNet module (same
  Agent B1 deliverable).
* **step 20** — ``assign_HMM_states_to_proxy_expr_vals`` /
  ``i3HMM_assign_HMM_states_to_proxy_expr_vals``
  (``inferCNV_ops.R:1463-1499``). State → representative expression
  value; strictly a viz/CN-ratio remap. **Ported inline** here (no
  separate module); dispatch by ``config.HMM_type``.
* **step 21** — ``mask_non_DE_genes_basic`` (``inferCNV_ops.R:1509-1552``).
  Guard: ``mask_nonDE_genes == TRUE``. Py:
  :func:`pyinfercnv.mask_de.wilcoxon.mask_non_DE_genes`.
* **step 22** — ``clear_noise_via_ref_mean_sd`` / ``clear_noise``
  (``inferCNV_ops.R:1560-1615``). Guard: ``denoise == TRUE``. Py:
  :func:`pyinfercnv.denoise.ref_mean_sd.denoise_by_ref_mean_sd`.

Ordering is **load-bearing**: R runs BayesNet → filter → repr-assign →
mask → denoise. BayesNet's output overrides HMM states in step 19, and
the mask/denoise operate on the post-overwrite matrix. Agents B1/B2/B3
must respect this order when they land actual implementations.

Current behaviour
-----------------
* When neither HMM states nor any Phase 3 toggle (``BayesMaxPNormal > 0``,
  ``mask_nonDE_genes``, ``denoise``) is active, returns ``result_phase2``
  unchanged.
* When HMM states are present, ``hmm_proxy_matrix`` is populated by step 20
  (R-faithful — R runs step 20 whenever ``HMM=TRUE``,
  ``inferCNV_ops.R:1463-1499``).
* When ``BayesMaxPNormal > 0``, runs steps 18+19; ``bayes_posterior`` is
  persisted from ``gibbs_result['cnv_posterior']`` (``gibbs.py:269-276``).
* When ``mask_nonDE_genes=True``, runs step 21.
* When ``denoise=True``, runs step 22.
* ``config.reassignCNVs=True`` (R default) raises ``NotImplementedError``
  at the orchestrator boundary; the Python Gibbs port only implements the
  ``removeCNV``-only branch (``gibbs.py:186-190``).

See ``docs/superpowers/plans/2026-04-24-phase3-start.md`` for the full
design and ``docs/superpowers/plans/2026-04-25-phase3-wire-fix-execution.md``
for the wire-up history.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pyinfercnv.config import InferCNVConfig
    from pyinfercnv.result import InferCNVResult


# ---------------------------------------------------------------------------
# Step 20 — state → CN-ratio lookup maps
#
# R source (i6): inferCNV_HMM.R:1195-1200  (``assign_HMM_states_to_proxy_expr_vals``)
#   expr.data[== 1] <- 0    # state 1 (0-based: 0) → 0.0
#   expr.data[== 2] <- 0.5  # state 2 (0-based: 1) → 0.5
#   expr.data[== 3] <- 1    # state 3 (0-based: 2) → 1.0  (neutral)
#   expr.data[== 4] <- 1.5  # state 4 (0-based: 3) → 1.5
#   expr.data[== 5] <- 2    # state 5 (0-based: 4) → 2.0
#   expr.data[== 6] <- 3    # state 6 (0-based: 5) → 3.0
#
# R source (i3): inferCNV_i3HMM.R:409-411  (``i3HMM_assign_HMM_states_to_proxy_expr_vals``)
#   expr.data[== 1] <- 0.5  # state 1 (0-based: 0) → 0.5
#   expr.data[== 2] <- 1    # state 2 (0-based: 1) → 1.0  (neutral)
#   expr.data[== 3] <- 1.5  # state 3 (0-based: 2) → 1.5
#
# Python hmm_states are 0-based (see filter_high_p_normals.py:32-34).
# A numpy fancy-index lookup replaces R's sequential in-place overwrite;
# it is equivalent because R's original values are integer state labels
# whose substitution order is irrelevant (no value overlaps).
# ---------------------------------------------------------------------------

_I6_STATE_TO_CN: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64
)
_I3_STATE_TO_CN: np.ndarray = np.array(
    [0.5, 1.0, 1.5], dtype=np.float64
)


def _step20_assign_states_to_proxy_expr_vals(
    result: "InferCNVResult",
    config: "InferCNVConfig",
    *,
    filtered_states: "np.ndarray | None" = None,
) -> np.ndarray:
    """Apply state→CN-ratio remap. Pure lookup: output[c,g] = map[state[c,g]].

    Parameters
    ----------
    result : InferCNVResult
        Phase 2 result carrying ``hmm_states`` (i6) or ``hmm_states_i3`` (i3).
    config : InferCNVConfig
        Only ``HMM_type`` is consumed.
    filtered_states : optional
        Post-step19 filtered matrix (from ``filter_high_p_normals``). If
        None, use the raw phase-2 matrix.

    Returns
    -------
    np.ndarray
        (n_cells, n_genes) float64 — state→CN ratio matrix matching
        R's ``hmm.infercnv_obj@expr.data`` after step 20.
    """
    hmm_type = config.HMM_type  # "i6" or "i3"

    if filtered_states is not None:
        states = np.asarray(filtered_states, dtype=np.int8)
    elif hmm_type == "i6":
        if result.hmm_states is None:
            raise ValueError(
                "_step20: config.HMM_type='i6' but result.hmm_states is None"
            )
        states = np.asarray(result.hmm_states, dtype=np.int8)
    elif hmm_type == "i3":
        if result.hmm_states_i3 is None:
            raise ValueError(
                "_step20: config.HMM_type='i3' but result.hmm_states_i3 is None"
            )
        states = np.asarray(result.hmm_states_i3, dtype=np.int8)
    else:
        raise ValueError(f"_step20: unknown HMM_type {hmm_type!r}; expected 'i6' or 'i3'")

    state_map = _I6_STATE_TO_CN if hmm_type == "i6" else _I3_STATE_TO_CN
    n_states = len(state_map)

    # Validate range to give a clear error instead of a silent wrap.
    s_min = int(states.min())
    s_max = int(states.max())
    if s_min < 0 or s_max >= n_states:
        raise ValueError(
            f"_step20: state values out of 0-based range [0, {n_states - 1}]: "
            f"min={s_min}, max={s_max}"
        )

    # Vectorised lookup — equivalent to R's sequential in-place overwrites.
    proxy = state_map[states.astype(np.intp)]  # (n_cells, n_genes) float64
    return proxy


def run_phase3(
    result_phase2: "InferCNVResult",
    *,
    config: "InferCNVConfig",
) -> "InferCNVResult":
    """Run Phase 3 post-processing (BayesNet / mask_non_DE / denoise).

    Parameters
    ----------
    result_phase2
        Output of :func:`pyinfercnv.pipeline_phase2.run_phase2`. Must
        carry ``hmm_states`` (i6) or ``hmm_states_i3`` (i3),
        ``cnv_regions``, and ``cnv_matrix`` / ``cnv_matrix_fc``.
    config
        :class:`pyinfercnv.config.InferCNVConfig`. Phase 3 toggles
        consulted here:

        * ``BayesMaxPNormal`` (>0 ⇒ step 18+19 fire; R:
          ``inferCNV_ops.R:275``).
        * ``mask_nonDE_genes`` (default False; R:
          ``inferCNV_ops.R:329``).
        * ``denoise`` (default False; R: ``inferCNV_ops.R:301``).

    Returns
    -------
    InferCNVResult
        ``result_phase2`` with Phase 3 fields populated where applicable:
        ``bayes_posterior`` (when ``BayesMaxPNormal > 0``), ``de_mask``
        (when ``mask_nonDE_genes`` is True), ``denoised_matrix`` (when
        ``denoise`` is True), and ``hmm_proxy_matrix`` (whenever HMM
        states are present, per R's step 20 policy at
        ``inferCNV_ops.R:1463-1499``). Returns ``result_phase2``
        unchanged when neither HMM states nor any Phase 3 toggle is
        active.

    Raises
    ------
    NotImplementedError
        When ``config.reassignCNVs=True`` (the R default) — the Python
        Gibbs port only implements the ``removeCNV``-only branch
        (``gibbs.py:186-190``). Set ``reassignCNVs=False`` on the
        config until the reassign branch is ported.
    KeyError
        When ``_step18_bayesnet`` returns a result dict that does not
        contain the canonical ``cnv_posterior`` key — surfaces a
        contract drift instead of silently leaving ``bayes_posterior``
        unset.
    """
    # Detect whether Phase 3 has any work to do. R-faithful policy:
    # step 20 (state → CN-ratio proxy) runs whenever HMM states exist
    # (R: ``inferCNV_ops.R:1463-1499``), independent of Bayes/mask/denoise.
    # Therefore HMM-only callers must pass through this body so step 20
    # populates ``hmm_proxy_matrix``; only when neither HMM states nor any
    # Phase 3 toggle is active do we short-circuit.
    bayes_on = config.BayesMaxPNormal > 0.0
    mask_on = bool(getattr(config, "mask_nonDE_genes", False))
    denoise_on = bool(config.denoise)

    has_i6_states = config.HMM_type == "i6" and result_phase2.hmm_states is not None
    has_i3_states = config.HMM_type == "i3" and result_phase2.hmm_states_i3 is not None
    has_hmm = has_i6_states or has_i3_states

    if not (bayes_on or mask_on or denoise_on or has_hmm):
        return result_phase2

    result = result_phase2
    filtered_states: np.ndarray | None = None

    # Step 18 + 19: BayesNet Gibbs posterior + filter_high_p_normals.
    # Requires Phase 2 to have persisted per-state (mu, sigma) via
    # hspike_calibration (i6) or i3_state_mus/sigmas (i3).
    if bayes_on:
        from pyinfercnv.bayesnet import _step18_bayesnet

        if config.HMM_type == "i6":
            if result.hspike_calibration is None:
                raise ValueError(
                    "run_phase3: HMM_type='i6' and BayesMaxPNormal>0 but "
                    "result.hspike_calibration is None. Did run_phase2 run "
                    "with HMM_type='i6'? The field is populated by "
                    "pipeline_phase2.run_phase2 after i6 hspike calibration."
                )
            gibbs_result, filtered_states = _step18_bayesnet(
                result, config,
                i6_calibration=result.hspike_calibration,
            )
        else:  # i3
            if result.i3_state_mus is None or result.i3_state_sigmas is None:
                raise ValueError(
                    "run_phase3: HMM_type='i3' and BayesMaxPNormal>0 but "
                    "result.i3_state_mus/i3_state_sigmas are None. Did "
                    "run_phase2 run with HMM_type='i3'?"
                )
            gibbs_result, filtered_states = _step18_bayesnet(
                result, config,
                i3_mus=result.i3_state_mus,
                i3_sigmas=result.i3_state_sigmas,
            )

        # Persist per-region per-state posterior for downstream inspection.
        # The canonical key is ``cnv_posterior`` (see ``gibbs.py:269-276``);
        # ``theta_mean`` is only a local variable inside the sampler. Fail
        # loudly on contract drift instead of silently leaving the field None.
        try:
            result.bayes_posterior = gibbs_result["cnv_posterior"]
        except (KeyError, TypeError) as exc:
            got = (
                f"keys={list(gibbs_result.keys())}"
                if isinstance(gibbs_result, dict)
                else f"type={type(gibbs_result).__name__}"
            )
            raise KeyError(
                "run_phase3 expected _step18_bayesnet() to return "
                f"gibbs_result['cnv_posterior']; got {got}"
            ) from exc

    # Step 20: state→CN-ratio proxy matrix. R-faithful: runs whenever HMM
    # states exist, regardless of Bayes/mask/denoise (``inferCNV_ops.R:1463-1499``).
    # Uses filtered_states when BayesNet ran; else the raw Phase 2 HMM state
    # matrix. ``has_hmm`` was computed at the top of this function.
    if has_hmm:
        result.hmm_proxy_matrix = _step20_assign_states_to_proxy_expr_vals(
            result, config, filtered_states=filtered_states,
        )

    # Step 21: mask non-DE genes
    if mask_on:
        from pyinfercnv.mask_de import _step21_mask_non_DE
        # adata is unused in the current implementation (delegates to
        # result.cell_meta + result.subclusters). Passed as None.
        result = _step21_mask_non_DE(result, None, config)

    # Step 22: denoise by ref mean ± sd
    if denoise_on:
        from pyinfercnv.denoise import _step22_denoise
        result = _step22_denoise(result, config)

    return result


__all__ = ["run_phase3", "_step20_assign_states_to_proxy_expr_vals"]

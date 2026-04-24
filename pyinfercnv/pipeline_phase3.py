"""Phase 3 top-level orchestrator — BayesNet + mask_non_DE + denoise.

Mirrors the shape of :func:`pyinfercnv.pipeline_phase2.run_phase2` but
runs after HMM state calls. Composes (not replaces) Phase 2 output.
Canonical composition::

    result_p1 = infercnv(adata, config=cfg, inplace=False, ...)
    result_p2 = run_phase2(result_p1, adata, config=cfg, ...)
    result    = run_phase3(result_p2, config=cfg, ...)

Or, once integrated into the top-level :func:`pyinfercnv.pipeline.infercnv`
gate (**not done in this skeleton**), the pipeline auto-invokes
``run_phase3`` when any of the three Phase 3 toggles fire.

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

Phase 3 skeleton contract (this commit)
---------------------------------------
* **No-op** when all three toggles are off (``BayesMaxPNormal == 0``
  AND ``mask_nonDE_genes == False`` AND ``denoise == False``). Returns
  ``result_phase2`` unchanged, preserving backward compatibility with
  Phase-2-only callers.
* **Raises NotImplementedError** when any toggle is on. Loud failure
  is deliberate: Phase 3 opt-in paths must not silently return
  neutral/unmasked states. The error message cites the responsible
  module so the second-round agent gets a clear handoff.

See ``docs/superpowers/plans/2026-04-24-phase3-start.md`` for the full
design, validation criteria, and Agent B1/B2/B3 task decomposition.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyinfercnv.config import InferCNVConfig
    from pyinfercnv.result import InferCNVResult


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
          ``inferCNV_ops.R:275``). **Default 0.5 — if run_phase3 is
          invoked with default config this branch raises.**
        * ``mask_nonDE_genes`` (default False; R:
          ``inferCNV_ops.R:329``).
        * ``denoise`` (default False; R: ``inferCNV_ops.R:301``).

    Returns
    -------
    InferCNVResult
        A *new* result with (eventually) populated Phase 3 fields:
        ``bayes_posterior``, ``de_mask``, ``denoised_matrix``. Today
        this is ``result_phase2`` unchanged in the all-toggles-off
        path; otherwise the function raises.

    Raises
    ------
    NotImplementedError
        Whenever any Phase 3 toggle is active. The message names the
        responsible module so the next agent can pick up.
    """
    # Short-circuit: all toggles off ⇒ Phase 3 is a no-op.
    bayes_on = config.BayesMaxPNormal > 0.0
    mask_on = bool(getattr(config, "mask_nonDE_genes", False))
    denoise_on = bool(config.denoise)

    if not (bayes_on or mask_on or denoise_on):
        return result_phase2

    # BayesNet (step 18+19) requires per-state (mu, sigma) from the Phase 2
    # hspike calibration (for i6) or estimate_i3_state_params (for i3). Phase
    # 2 currently keeps these local; InferCNVResult does not persist them.
    # Fail fast to keep users from silently skipping step 18+19.
    if bayes_on:
        raise NotImplementedError(
            "Phase 3 BayesNet orchestrator gate: _step18_bayesnet requires "
            "i6_calibration=HspikeCalibration (HMM_type='i6') or i3_mus/"
            "i3_sigmas (HMM_type='i3'), which pipeline_phase2 does not yet "
            "persist onto InferCNVResult. The module "
            "pyinfercnv.bayesnet._step18_bayesnet is independently tested "
            "via tests/test_r_parity.py::test_step18_bayesnet_parity. Wire "
            "in a follow-up commit after adding an hspike_calibration field "
            "to InferCNVResult. To use BayesNet directly, call "
            "run_bayesnet_gibbs / filter_high_p_normals from pyinfercnv.bayesnet."
        )

    # From here on at least one of (mask, denoise) is on — both are wired.
    result = result_phase2

    if mask_on:
        from pyinfercnv.mask_de import _step21_mask_non_DE
        # adata is unused in the current implementation (delegates to
        # result.cell_meta + result.subclusters). Passed as None.
        result = _step21_mask_non_DE(result, None, config)

    if denoise_on:
        from pyinfercnv.denoise import _step22_denoise
        result = _step22_denoise(result, config)

    return result


__all__ = ["run_phase3"]

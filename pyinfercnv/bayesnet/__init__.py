"""Phase 3 — BayesNet Gibbs posterior of HMM states (R step 18+19).

R source: ``R/inferCNV_BayesNet.R`` (rjags BUGS Mixture Model) plus
``inst/BUGS_Mixture_Model`` (i6) / ``inst/BUGS_Mixture_Model_i3`` (i3).

Public entry points
-------------------
:func:`run_bayesnet_gibbs`
    Numba Gibbs sampler producing per-region posterior probabilities
    (R step 18).
:func:`filter_high_p_normals`
    Overwrite high-P(normal) regions back to neutral in the HMM state
    matrix (R step 19, ``removeCNV`` branch).
:func:`_step18_bayesnet`
    Convenience wrapper used by :mod:`pyinfercnv.pipeline_phase3` that
    runs steps 18+19 back-to-back and returns ``(bayesnet_result,
    filtered_states)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinfercnv.bayesnet.filter_high_p_normals import filter_high_p_normals
from pyinfercnv.bayesnet.gibbs import run_bayesnet_gibbs

if TYPE_CHECKING:  # pragma: no cover
    from pyinfercnv.config import InferCNVConfig
    from pyinfercnv.hmm.hspike import HspikeCalibration
    from pyinfercnv.result import InferCNVResult


def _step18_bayesnet(
    result: "InferCNVResult",
    config: "InferCNVConfig",
    *,
    i6_calibration: "HspikeCalibration | None" = None,
    i3_mus: np.ndarray | None = None,
    i3_sigmas: np.ndarray | None = None,
    numBurnin: int = 1000,  # noqa: N803
    numSamples: int = 1000,  # noqa: N803
    numChains: int = 3,  # noqa: N803
) -> tuple[dict, np.ndarray]:
    """Run R step 18 + 19 on a Phase-2 result; return (gibbs_result, filtered_states).

    Called by :mod:`pyinfercnv.pipeline_phase3` when ``BayesMaxPNormal > 0``.
    The caller is responsible for supplying per-state (mu, sigma) —
    Phase 2 already computes these to drive the HMM emission.

    Parameters
    ----------
    result
        Phase-2 :class:`InferCNVResult` carrying ``cnv_matrix_fc``,
        ``subclusters``, ``hmm_states`` (i6) or ``hmm_states_i3`` (i3),
        and ``cnv_regions``.
    config
        :class:`InferCNVConfig`. Consumed fields: ``HMM_type``,
        ``BayesMaxPNormal``, ``random_state``.
    i6_calibration
        Required when ``HMM_type='i6'``; the Phase 2 hspike calibration.
    i3_mus, i3_sigmas
        Required when ``HMM_type='i3'``; per-state params from
        :func:`pyinfercnv.hmm.i3.estimate_i3_state_params`.
    numBurnin, numSamples, numChains
        Gibbs knobs; see :func:`run_bayesnet_gibbs`.

    Returns
    -------
    gibbs_result : dict
        As returned by :func:`run_bayesnet_gibbs`.
    filtered_states : np.ndarray
        ``(n_cells, n_bins)`` int8 — the input HMM state matrix with
        high-P(normal) regions overwritten to neutral.
    """
    hmm_type = config.HMM_type
    if hmm_type == "i6":
        if i6_calibration is None:
            raise ValueError("_step18_bayesnet: HMM_type='i6' requires i6_calibration")
        state_mus = np.asarray(i6_calibration.state_mus, dtype=np.float64)
        # Use num_cells=1 as the baseline; per-subcluster sigmas would vary by
        # region, which would break the single-sigma BUGS model contract. R's
        # MeanSD uses the raw spike sd (inferCNV_BayesNet.R:156-161) without
        # per-subcluster scaling, so we follow R here.
        state_sigmas = np.asarray(i6_calibration.state_sigmas, dtype=np.float64)
        hmm_state_matrix = result.hmm_states
    else:  # i3
        if i3_mus is None or i3_sigmas is None:
            raise ValueError("_step18_bayesnet: HMM_type='i3' requires i3_mus and i3_sigmas")
        state_mus = np.asarray(i3_mus, dtype=np.float64)
        state_sigmas = np.asarray(i3_sigmas, dtype=np.float64)
        hmm_state_matrix = result.hmm_states_i3

    if hmm_state_matrix is None:
        raise ValueError(
            f"_step18_bayesnet: result lacks HMM state matrix for HMM_type={hmm_type!r}"
        )
    if result.cnv_regions is None:
        raise ValueError("_step18_bayesnet: result.cnv_regions is None")
    if result.subclusters is None:
        raise ValueError("_step18_bayesnet: result.subclusters is None")

    gibbs_result = run_bayesnet_gibbs(
        hmm_state_matrix=np.asarray(hmm_state_matrix),
        cnv_matrix=np.asarray(result.cnv_matrix_fc),
        cnv_regions=result.cnv_regions,
        state_mus=state_mus,
        state_sigmas=state_sigmas,
        subclusters=np.asarray(result.subclusters),
        HMM_type=hmm_type,
        BayesMaxPNormal=config.BayesMaxPNormal,
        numBurnin=numBurnin,
        numSamples=numSamples,
        numChains=numChains,
        random_state=config.random_state,
    )

    filtered_states = filter_high_p_normals(
        np.asarray(hmm_state_matrix),
        gibbs_result,
        HMM_type=hmm_type,
        BayesMaxPNormal=config.BayesMaxPNormal,
    )

    return gibbs_result, filtered_states


__all__ = [
    "run_bayesnet_gibbs",
    "filter_high_p_normals",
    "_step18_bayesnet",
]

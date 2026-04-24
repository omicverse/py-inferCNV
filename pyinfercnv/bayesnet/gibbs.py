"""Gibbs sampler for HMM state posterior — R-parity port of inferCNVBayesNet.

R source map
------------
* ``R/inferCNV_BayesNet.R:1054-1107`` — ``run_gibb_sampling`` (delegates to
  ``rjags::jags.model`` + ``rjags::coda.samples``; JAGS model loaded from
  ``inferCNV_BayesNet.R:1285-1287``).
* ``inst/BUGS_Mixture_Model`` — i6 six-state mixture:
  ``gexp[i,j] ~ dnorm(mu.1[j], tau.1[j])`` with state-mixture mean /
  precision, ``epsilon[j] ~ dcat(theta[])``, ``theta ~ ddirich(1..1)``.
* ``inst/BUGS_Mixture_Model_i3`` — i3 three-state variant (same structure).
* ``R/inferCNV_BayesNet.R:1237-1364`` — ``inferCNVBayesNet`` orchestration.
* ``R/inferCNV_BayesNet.R:1137-1150`` — ``cnv_prob`` / ``cell_prob``
  posterior extraction.
* ``R/inferCNV_ops.R:1362-1393`` — pipeline wiring (step 18, HMM==TRUE +
  ``BayesMaxPNormal > 0``).

Python-port design decisions (Agent B1, 2026-04-24)
---------------------------------------------------
* **Mu / sigma sourcing.** R's ``MeanSD`` reads per-state (mu, sigma)
  from the hspike object (``inferCNV_BayesNet.R:154``); we take
  ``state_mus`` / ``state_sigmas`` as explicit kwargs supplied by the
  caller (typically ``HspikeCalibration.state_mus`` and
  ``HspikeCalibration.sigmas_for_num_cells(n)``). R stores ``1/sigma^2``
  in ``obj@sig`` because JAGS uses precision; Python keeps ``sd`` and
  converts internally.
* **Per-region granularity.** Each row of the ``cnv_regions`` DataFrame
  (output of ``pipeline_phase2._build_cnv_regions``) is one CNV: a
  (subcluster × chromosome × non-neutral state run) block. Gibbs runs
  independently per region (R ``nonParallel`` / ``withParallel``). For
  each region, ``Cells`` = ``np.where(subclusters == sc)[0]`` and
  ``Genes`` = bins ``bin_start..bin_end`` inclusive.
* **Matrix space.** Gibbs ingests **linear-FC** CNV values
  (``result.cnv_matrix_fc``, post-step14 invert_log2 + step16 outlier
  prune). Hspike ``state_mus`` / ``state_sigmas`` are in the same space.
  Callers that pass log2 values will see shifted posteriors; the
  pipeline wrapper is responsible for the right dtype.
* **Chain count.** R hardcodes ``n.chains = 6`` for i6 and ``3`` for i3
  (``inferCNV_BayesNet.R:1098``). This port defaults to 3 for both
  HMM types to keep the oligo fixture wall time inside the 60-s budget
  for Phase 3. ``numChains`` kwarg lets callers opt into R's 6/i6 value.
* **RNG non-equivalence.** R's JAGS MT19937 / L'Ecuyer streams and
  numba's PCG-64 differ; ``|ΔP|`` parity is the target, not trace
  reproduction. Seed is controlled via ``random_state``.

Non-goals (punted to follow-up sessions)
----------------------------------------
* ``reassignCNV`` branch (``inferCNV_BayesNet.R:491-540``) — R default
  ``reassignCNVs=TRUE`` also overwrites non-normal-majority regions to
  their argmax state after the ``removeCNV`` pass. This port implements
  ``removeCNV`` only; callers that want reassign must layer it on.
* ``removeCells`` branch — R's ``postMcmcMethod='removeCells'`` is not
  the default; skipped until a downstream demand surfaces.
* BayesNet diagnostic plots (``plotProbabilities`` / ``postProbNormal``)
  — viz is Phase 3b.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinfercnv.kernels.bayesnet_gibbs_numba import gibbs_sample_regions, pack_regions

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# --------------------------------------------------------------------------- #
# Public entry                                                                #
# --------------------------------------------------------------------------- #


def run_bayesnet_gibbs(
    hmm_state_matrix: np.ndarray,
    cnv_matrix: np.ndarray,
    cnv_regions: "pd.DataFrame",
    *,
    state_mus: np.ndarray,
    state_sigmas: np.ndarray,
    subclusters: np.ndarray,
    HMM_type: str = "i6",  # noqa: N803
    BayesMaxPNormal: float = 0.5,  # noqa: N803
    numBurnin: int = 1000,  # noqa: N803
    numSamples: int = 1000,  # noqa: N803
    numChains: int = 3,  # noqa: N803
    postProbsToRemoveLimit: float = 0.5,  # noqa: N803 — R-task-book alias; see Note
    reassignCNVs: bool = False,  # noqa: N803 — R default TRUE but not ported
    CORES: int = 1,  # noqa: N803
    quietly: bool = True,
    diagnostics: bool = False,
    postMcmcMethod: str | None = "removeCNV",  # noqa: N803
    random_state: int = 42,
) -> dict:
    """Run Bayesian Network Gibbs sampler over HMM-predicted CNV regions.

    Parameters
    ----------
    hmm_state_matrix
        Int-typed HMM state assignments from Phase 2, shape
        ``(n_cells, n_bins)``. Values ``0..K-1`` where ``K=6`` for i6 and
        ``3`` for i3; Python uses 0-based indices (R uses 1-based, so
        R's neutral state 3 ↔ Python index 2).
    cnv_matrix
        Phase-1/2 CNV matrix in **linear-FC** space (``result.cnv_matrix_fc``),
        ``(n_cells, n_bins)`` float32/float64. Must match the space in which
        ``state_mus`` / ``state_sigmas`` were calibrated.
    cnv_regions
        Output of :func:`pyinfercnv.pipeline_phase2._build_cnv_regions`.
        Each row defines one CNV block: columns
        ``(cell_group, subcluster, chromosome, bin_start, bin_end, state, cn)``.
    state_mus, state_sigmas
        Shape ``(K,)`` float64 arrays. For i6 these come from
        :attr:`HspikeCalibration.state_mus` and
        ``HspikeCalibration.sigmas_for_num_cells(n_members)``; for i3 from
        :func:`pyinfercnv.hmm.i3.estimate_i3_state_params`. Caller is
        responsible for using subcluster-size-scaled sigmas per region
        or a global ``sigmas_for_num_cells(1)`` approximation — see Note.
    subclusters
        Shape ``(n_cells,)`` int — per-cell subcluster label.
        ``pipeline_phase2._SUBCLUSTER_REF_SENTINEL`` identifies reference
        cells and those are excluded from all regions.
    HMM_type
        ``"i6"`` or ``"i3"``. Determines K (6 or 3) and the neutral state
        index used by downstream :mod:`pyinfercnv.bayesnet.filter_high_p_normals`.
    BayesMaxPNormal
        R ``inferCNV_ops.R:275``. Probability threshold for dropping CNVs
        whose posterior P(normal) exceeds this value. Not applied inside
        ``run_bayesnet_gibbs`` itself; consumed by
        :func:`filter_high_p_normals`. Pass-through for signature parity.
    numBurnin, numSamples
        Gibbs burn-in / main-sampling iterations **per chain**. R's JAGS
        uses ``n.adapt=500``, ``update(200)`` (warm-up), ``n.iter=1000``
        (``inferCNV_BayesNet.R:1099-1105``); the port combines
        adapt+update into ``numBurnin`` (default 1000). Increase to
        ``numBurnin=700, numSamples=1000`` for stricter R-parity.
    numChains
        Parallel chains per region. Default 3 (runtime budget); set to 6
        for R-parity on i6.
    postProbsToRemoveLimit
        R-task-book alias. **Not in R source** — kept as a no-op
        passthrough for the Agent-B1 contract declared in the task brief.
        Use :func:`filter_high_p_normals` with explicit ``BayesMaxPNormal``
        for the actual filter logic.
    reassignCNVs
        R default ``TRUE`` (``inferCNV_BayesNet.R:1306``). This port
        defaults to **False** — the reassign branch is not implemented.
        Passing ``True`` currently raises ``NotImplementedError`` from
        :func:`filter_high_p_normals`; keep the default.
    CORES, quietly, diagnostics, postMcmcMethod
        Pass-through for R signature parity. No behavioural effect here
        (Gibbs parallelism is via numba ``prange`` over regions, not
        ``CORES``).
    random_state
        Seed for the numpy PCG-64 generator that derives per-region
        Gibbs seeds. **Not bit-equivalent to R's JAGS MT19937 stream**;
        parity target is posterior probability.

    Returns
    -------
    dict
        ``{"cnv_posterior":         (n_regions, K) float64,
           "cell_posterior_counts": (sum_r C_r, K) int64,
           "cell_ptr":              (n_regions+1,) int64,
           "region_cell_idx":       list[np.ndarray]  — global cell indices per region,
           "region_gene_idx":       list[np.ndarray]  — global bin indices per region,
           "num_recorded_samples":  int (numChains * numSamples)}``

        ``cnv_posterior[r, k]`` is R ``colMeans(cnv_prob[[r]])[k]`` — the
        posterior mean of ``theta[k]`` for region ``r``. Divide
        ``cell_posterior_counts`` by ``num_recorded_samples`` to get the
        per-cell R ``cell_probabilities`` marginal.
    """
    if HMM_type not in ("i6", "i3"):
        raise ValueError(f"HMM_type must be 'i6' or 'i3', got {HMM_type!r}")
    if postMcmcMethod not in (None, "removeCNV", "removeCells"):
        raise ValueError(
            f"postMcmcMethod must be None|'removeCNV'|'removeCells', "
            f"got {postMcmcMethod!r}"
        )
    if postMcmcMethod == "removeCells":
        raise NotImplementedError(
            "postMcmcMethod='removeCells' not ported in this agent's scope; "
            "use 'removeCNV' (R default)."
        )
    if reassignCNVs:
        raise NotImplementedError(
            "reassignCNVs=True is R's default but not ported in this agent's "
            "scope (inferCNV_BayesNet.R:491-540). Pass reassignCNVs=False."
        )
    if numChains <= 0 or numBurnin < 0 or numSamples <= 0:
        raise ValueError(
            f"numChains / numBurnin / numSamples invalid: "
            f"{numChains}, {numBurnin}, {numSamples}"
        )

    state_mus = np.ascontiguousarray(state_mus, dtype=np.float64)
    state_sigmas = np.ascontiguousarray(state_sigmas, dtype=np.float64)

    K_expected = 6 if HMM_type == "i6" else 3
    if state_mus.shape != (K_expected,) or state_sigmas.shape != (K_expected,):
        raise ValueError(
            f"state_mus / state_sigmas must have shape ({K_expected},) for "
            f"HMM_type={HMM_type!r}; got {state_mus.shape} / {state_sigmas.shape}"
        )
    if np.any(state_sigmas <= 0.0):
        raise ValueError(
            f"state_sigmas must be > 0; got {state_sigmas}"
        )

    cnv_matrix_f64 = np.ascontiguousarray(cnv_matrix, dtype=np.float64)
    subclusters_arr = np.asarray(subclusters)

    n_regions = len(cnv_regions)

    # Empty-regions shortcut
    if n_regions == 0:
        return {
            "cnv_posterior": np.zeros((0, K_expected), dtype=np.float64),
            "cell_posterior_counts": np.zeros((0, K_expected), dtype=np.int64),
            "cell_ptr": np.zeros(1, dtype=np.int64),
            "region_cell_idx": [],
            "region_gene_idx": [],
            "num_recorded_samples": int(numChains) * int(numSamples),
        }

    # --- Assemble per-region gexp submatrices (shape (G_r, C_r)) ---
    gexps: list[np.ndarray] = []
    region_cell_idx: list[np.ndarray] = []
    region_gene_idx: list[np.ndarray] = []

    for row in cnv_regions.itertuples(index=False):
        sc = int(getattr(row, "subcluster"))
        bin_start = int(getattr(row, "bin_start"))
        bin_end = int(getattr(row, "bin_end"))  # R-parity: inclusive
        cell_idx = np.where(subclusters_arr == sc)[0].astype(np.intp)
        gene_idx = np.arange(bin_start, bin_end + 1, dtype=np.intp)

        if cell_idx.size == 0 or gene_idx.size == 0:
            # Preserve region positional alignment with placeholder (G=0 or C=0)
            gexps.append(np.zeros((max(gene_idx.size, 0), max(cell_idx.size, 0)), dtype=np.float64))
        else:
            sub = cnv_matrix_f64[np.ix_(cell_idx, gene_idx)]  # (C_r, G_r)
            gexps.append(np.ascontiguousarray(sub.T))  # want (G_r, C_r)

        region_cell_idx.append(cell_idx)
        region_gene_idx.append(gene_idx)

    gexp_flat, region_ptr, region_gc = pack_regions(gexps)

    # Per-region seeds (derived deterministically from random_state)
    seed_rng = np.random.default_rng(int(random_state))
    seeds = seed_rng.integers(1, 2**31 - 1, size=n_regions, dtype=np.int64)

    theta_mean, epsilon_counts, cell_ptr = gibbs_sample_regions(
        gexp_flat,
        region_ptr,
        region_gc,
        state_mus,
        state_sigmas,
        seeds,
        int(numBurnin),
        int(numSamples),
        int(numChains),
    )

    num_recorded = int(numChains) * int(numSamples)

    return {
        "cnv_posterior": np.asarray(theta_mean, dtype=np.float64),
        "cell_posterior_counts": np.asarray(epsilon_counts, dtype=np.int64),
        "cell_ptr": np.asarray(cell_ptr, dtype=np.int64),
        "region_cell_idx": region_cell_idx,
        "region_gene_idx": region_gene_idx,
        "num_recorded_samples": num_recorded,
    }


__all__ = ["run_bayesnet_gibbs"]

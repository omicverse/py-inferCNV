"""Numba-accelerated Gibbs sampler for the BayesNet mixture model.

R source map
------------
This module is the Python-numba port of R's JAGS/BUGS mixture model at
``infercnv/inst/BUGS_Mixture_Model`` (i6) and ``BUGS_Mixture_Model_i3`` (i3),
driven by ``R/inferCNV_BayesNet.R::run_gibb_sampling`` (L1054-1107).

BUGS model
----------
For each CNV region independently (one region = one (subcluster, chromosome-
run) block from :func:`pyinfercnv.pipeline_phase2._build_cnv_regions`):

    gexp[i, j] ~ Normal(mu[epsilon[j]], sd[epsilon[j]])   # i..G genes, j..C cells
    epsilon[j] ~ Categorical(theta[1..K])
    theta[1..K] ~ Dirichlet(1, 1, ..., 1)                 # flat prior

where ``(mu, sd)`` are **fixed** per-state parameters (R's `.get_gene_expr_
mean_sd_by_cnv` output, i.e. the hspike calibration in linear-FC space; see
:class:`pyinfercnv.hmm.hspike.HspikeCalibration`). The BUGS file contains a
dummy ``sigma ~ dgamma(1,1)`` prior that is never referenced in the likelihood,
so Python can skip it.

Full conditionals
-----------------
* ``epsilon[j] | theta, gexp`` ~ Categorical with unnormalised log-weights::

      log w[j, k] = log theta[k]
                  + sum_i [ -0.5 * ((gexp[i,j] - mu[k]) / sd[k])**2 - log(sd[k]) ]

  The Gaussian log-density constant ``-0.5 * log(2*pi)`` cancels in the
  normalisation, so we drop it for numerical stability.
* ``theta | epsilon`` ~ Dirichlet(alpha + counts) where ``alpha[k] = 1`` and
  ``counts[k] = #{j : epsilon[j] == k}``. We sample via the standard
  ``gamma / sum`` identity.

Determinism / RNG
-----------------
The numba inner loop uses numpy's ``np.random`` state seeded by the caller.
Numba's RNG is PCG-64 internally (as of numba 0.58+) when called through
``np.random.seed(...)`` + ``np.random.*`` APIs inside ``@njit``. Each CNV
region is a disjoint sub-problem, so regions are parallelised via
``prange``; within a region, chains run serially and states update
sequentially (the Markov chain property).

**Not bit-equivalent to R**. R's JAGS uses a different RNG stream (L'Ecuyer
by default under ``coda``), and numba's RNG is not MT19937. The parity
target for :mod:`pyinfercnv.bayesnet` is the posterior probability
``|ΔP| < 0.05`` (tier 3.5 empirical), not sample-by-sample reproduction.

Memory layout
-------------
All inputs are contiguous float64 / int64 arrays. The kernel expects CNV
regions to be passed as a single long-format triple::

    gexp_flat   : (sum_r (G_r * C_r),) float64
    region_ptr  : (n_regions + 1,) int64   — CSR-style offset into gexp_flat
    region_gc   : (n_regions, 2) int64    — per-region (G_r, C_r)

This avoids jagged-array boxing inside ``@njit`` (numba rejects
``list[ndarray]`` with varying shapes in njit kernels).

Outputs
-------
``theta_mean``        : (n_regions, K) float64 — per-region Dirichlet posterior
                        mean (≡ R's ``colMeans(cnv_prob_samples)``).
``epsilon_counts``    : (sum_r C_r, K) int64   — per-cell state frequency over
                        saved samples (≡ R's ``cell_prob`` frequency table,
                        un-normalised; caller divides by ``num_samples``).
``cell_ptr``          : (n_regions + 1,) int64 — CSR offsets into
                        ``epsilon_counts`` per region (= ``region_gc[:, 1]``
                        cumsum).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- #
# Internal helpers (pure-python; also used by tests)                          #
# --------------------------------------------------------------------------- #


def pack_regions(
    gexps: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of ``(G_r, C_r)`` float64 arrays into CSR-style triple.

    The numba kernel requires contiguous arrays (njit can't box jagged
    ndarrays safely). Callers build ``gexps`` by slicing the full CNV
    matrix per region; this helper concatenates in order.
    """
    if not gexps:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
        )
    region_gc = np.array([[g.shape[0], g.shape[1]] for g in gexps], dtype=np.int64)
    sizes = region_gc[:, 0] * region_gc[:, 1]
    region_ptr = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    gexp_flat = np.concatenate([np.ascontiguousarray(g, dtype=np.float64).ravel(order="C") for g in gexps])
    return gexp_flat, region_ptr, region_gc


# --------------------------------------------------------------------------- #
# Numba inner loop                                                            #
# --------------------------------------------------------------------------- #


@njit(cache=True, parallel=True, fastmath=False)
def gibbs_sample_regions(
    gexp_flat: np.ndarray,        # (sum G_r*C_r,) float64
    region_ptr: np.ndarray,       # (n_regions+1,) int64
    region_gc: np.ndarray,        # (n_regions, 2) int64: (G_r, C_r)
    mus: np.ndarray,              # (K,) float64
    sds: np.ndarray,              # (K,) float64
    seeds: np.ndarray,            # (n_regions,) int64 — one seed per region
    num_burnin: int,
    num_samples: int,
    num_chains: int,
) -> tuple:                       # returns (theta_mean[n_regions,K], epsilon_counts[sum C_r, K])
    """Run Gibbs sampling independently over CNV regions; parallel over regions.

    Per region and per chain:
      1. Initialise ``epsilon[j] = chain_id`` for j in 1..C_r (R-parity,
         ``inferCNV_BayesNet.R:1078-1085``).
      2. Burn-in ``num_burnin`` iterations (no samples recorded).
      3. Main sampling ``num_samples`` iterations; accumulate ``theta`` mean
         and ``epsilon`` per-state counts.

    Total recorded samples per region = ``num_chains * num_samples``.
    """
    n_regions = region_gc.shape[0]
    K = mus.shape[0]

    # CSR offsets for per-cell epsilon counts across regions
    cell_ptr = np.zeros(n_regions + 1, dtype=np.int64)
    for r in range(n_regions):
        cell_ptr[r + 1] = cell_ptr[r] + region_gc[r, 1]
    total_cells = cell_ptr[n_regions]

    theta_mean = np.zeros((n_regions, K), dtype=np.float64)
    epsilon_counts = np.zeros((total_cells, K), dtype=np.int64)

    # Precompute log(sd) for each state (will need inside inner loop)
    log_sds = np.empty(K, dtype=np.float64)
    for k in range(K):
        log_sds[k] = math.log(sds[k])

    for r in prange(n_regions):
        G_r = region_gc[r, 0]
        C_r = region_gc[r, 1]
        if G_r == 0 or C_r == 0:
            continue

        # Slice the per-region gexp (G_r, C_r) as a flat buffer; we'll
        # index gexp[i*C_r + j] inside the loop.
        r_start = region_ptr[r]

        # Seed numba's RNG per region (independent streams across regions)
        np.random.seed(seeds[r])

        # ----- Precompute per-(j, k) gene-sum log-likelihood term -----
        #   ll_cell_state[j, k] = sum_i [-0.5 * ((gexp[i,j]-mu[k])/sd[k])^2 - log(sd[k])]
        # Shape (C_r, K). This is the only data-dependent quantity in the
        # categorical full-conditional; we compute once per region and reuse
        # across all chains and iterations.
        ll = np.empty((C_r, K), dtype=np.float64)
        for j in range(C_r):
            for k in range(K):
                acc = 0.0
                mu_k = mus[k]
                inv_sd = 1.0 / sds[k]
                for i in range(G_r):
                    x = gexp_flat[r_start + i * C_r + j]
                    z = (x - mu_k) * inv_sd
                    acc += -0.5 * z * z - log_sds[k]
                ll[j, k] = acc

        # Aggregators across chains × iterations
        local_theta_sum = np.zeros(K, dtype=np.float64)
        local_eps_counts = np.zeros((C_r, K), dtype=np.int64)
        total_iters_recorded = 0

        # Working buffers
        epsilon = np.empty(C_r, dtype=np.int64)
        theta = np.empty(K, dtype=np.float64)
        counts = np.empty(K, dtype=np.int64)
        log_w = np.empty(K, dtype=np.float64)
        cum = np.empty(K, dtype=np.float64)

        for chain in range(num_chains):
            # --- Init epsilon (R-parity: each chain initialised to its state id) ---
            init_state = chain % K
            for j in range(C_r):
                epsilon[j] = init_state

            # Initial theta from flat Dirichlet(1..1); sample via Gamma(1,1) = Exp(1)
            gsum = 0.0
            for k in range(K):
                # np.random.gamma(1.0, 1.0) = Exponential(1)
                gk = np.random.gamma(1.0, 1.0)
                theta[k] = gk
                gsum += gk
            for k in range(K):
                theta[k] = theta[k] / gsum

            total_iters = num_burnin + num_samples
            for it in range(total_iters):
                # --- Sample epsilon[j] | theta, gexp ---
                log_theta = np.empty(K, dtype=np.float64)
                for k in range(K):
                    # protect against log(0) from extreme draws
                    if theta[k] < 1e-300:
                        log_theta[k] = -690.0  # ~ log(1e-300)
                    else:
                        log_theta[k] = math.log(theta[k])

                for k in range(K):
                    counts[k] = 0

                for j in range(C_r):
                    # log_w[k] = log theta[k] + ll[j, k]
                    max_lw = -1e300
                    for k in range(K):
                        lw = log_theta[k] + ll[j, k]
                        log_w[k] = lw
                        if lw > max_lw:
                            max_lw = lw
                    # softmax via log-sum-exp then cumulative distribution
                    s = 0.0
                    for k in range(K):
                        w = math.exp(log_w[k] - max_lw)
                        cum[k] = w
                        s += w
                    # normalise and cumulative sum
                    c = 0.0
                    for k in range(K):
                        c += cum[k] / s
                        cum[k] = c
                    u = np.random.random()
                    # linear search (K ≤ 6 so branch-free binary search not worth it)
                    chosen = K - 1
                    for k in range(K):
                        if u <= cum[k]:
                            chosen = k
                            break
                    epsilon[j] = chosen
                    counts[chosen] += 1

                # --- Sample theta | epsilon ---
                gsum = 0.0
                for k in range(K):
                    # alpha = 1 prior (BUGS ddirich(alpha[]) with alpha[i]<-1)
                    shape = 1.0 + float(counts[k])
                    gk = np.random.gamma(shape, 1.0)
                    theta[k] = gk
                    gsum += gk
                for k in range(K):
                    theta[k] = theta[k] / gsum

                # --- Record after burn-in ---
                if it >= num_burnin:
                    for k in range(K):
                        local_theta_sum[k] += theta[k]
                    for j in range(C_r):
                        local_eps_counts[j, epsilon[j]] += 1
                    total_iters_recorded += 1

        # Write reductions into the output arrays
        inv_n = 1.0 / float(total_iters_recorded) if total_iters_recorded > 0 else 0.0
        for k in range(K):
            theta_mean[r, k] = local_theta_sum[k] * inv_n
        cell_off = cell_ptr[r]
        for j in range(C_r):
            for k in range(K):
                epsilon_counts[cell_off + j, k] = local_eps_counts[j, k]

    return theta_mean, epsilon_counts, cell_ptr


__all__ = ["pack_regions", "gibbs_sample_regions"]

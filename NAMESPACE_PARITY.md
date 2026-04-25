# NAMESPACE_PARITY — R infercnv → pyinfercnv

Per `docs/superpowers/specs/2026-04-21-pyinfercnv-master-design.md` §3.3 + §5.2.

Tier meanings (spec §5.1):
- **4 bit-exact**   `max_diff < 1e-10`
- **4 approximate** `max_diff < 1e-6`
- **4 (relaxed)**   `max_diff < 1e-3` (float32 cumulative)
- **3.5 empirical** stochastic — Spearman ρ / Jaccard floor
- **3** identity    same ID set / labels

**Primary parity metric (Phase 2 and on): Spearman ρ on the continuous
post-Phase-1 CNV matrix (step 14).** Measurements (cnv_matrix_spearman.md):
oligo 0.9998; DCIS1 / TNBC1 / TNBC3 all 1.0000. Subcluster ARI on Leiden
output is **operational only**, not a parity claim — bucket IDs are
arbitrary algorithm-internal labels with no py-vs-R semantic contract
(Jason's v7 review; see HANDOFF.md §2.1).

## Phase 1 (R steps 1–14 + 16)

| R source / function | Python submodule | Phase | Tier | Status | R-parity metric |
|---|---|---|---|---|---|
| `CreateInfercnvObject` (chr_exclude, gene_order) | `pipeline._build_chromosome_layout` | 1 | 3 | ✓ | same gene set |
| `require_above_min_mean_expr_cutoff` | `preprocess.filter_low_expression_genes` | 1 | **4 bit-exact** | ✓ | set + value equality |
| `normalize_counts_by_seq_depth` | `preprocess.normalize_by_seq_depth` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 path, 2026-04-23) |
| `log2xplus1` | `preprocess.log2_plus1` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 log1p) |
| `subtract_ref_expr_from_obs` (use_bounds=TRUE, 1st pass) | `preprocess.subtract_reference` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 mean + bounds) |
| `apply_max_threshold_bounds` | `preprocess.apply_max_centered_threshold` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 clip) |
| `.smooth_center_helper` + `.smooth_helper` (`smooth_by_chromosome`) | `smooth.smooth_pyramidinal` + `kernels.smooth_tail_overwrite` | 1 | 4 approximate | ✓ | float64 interior + tail; empirical ≤ 1e-3 (scipy `uniform_filter1d` accumulation order vs R) |
| `center_cell_expr_across_chromosome` (median) | `center.center_cells` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 median) |
| `subtract_ref_expr_from_obs` (2nd pass) | `preprocess.subtract_reference` | 1 | **4 bit-exact** | ✓ | covered by step12 parity |
| `invert_log2` | `preprocess.invert_log2` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 exp2) |
| `remove_outliers_norm` (`average_bound`) | `cna.prune_outliers` | 1 | **4 bit-exact** | ✓ | `max_diff < 1e-10` (float64 bounds + clip) |

## Phase 2 (R steps 15, 17)

| R source / function | Python submodule | Phase | Tier | Status | R-parity metric |
|---|---|---|---|---|---|
| `define_signif_tumor_subclusters_via_leiden` | `subcluster.leiden.leiden_subcluster` | 2 | 3.5 empirical | ✓ | **Primary**: CNV-matrix Spearman ρ = 1.0000 on DCIS1/TNBC1/TNBC3, 0.9998 on oligo (upstream step14 is what this subcluster consumes). Subcluster ARI kept as operational floor 0.85 (`tests/test_r_parity.py::test_step15_subclusters_ari_floor`) — not a parity claim; see HANDOFF.md §2.1 |
| `define_signif_tumor_subclusters_via_random_smooothed_trees` | `subcluster.random_trees` | 2 | 3.5 empirical | ✓ | implemented; selected via `tumor_subcluster_partition_method="random_trees"` |
| `.qnorm` (quantile normalisation helper) | `subcluster.qnorm` | 2 | 4 approximate | ✓ | per-column rank-based; exercised as part of random_trees |
| `inferCNV_i3HMM` (`predict_i3`) | `hmm.predict_i3` | 2 | 3.5 empirical | ✓ | Jaccard 1.000 vs R step17_hmm_i3 (floor 0.99, spec §5.2 target 0.95) |
| `inferCNV_HMM` (i6, `predict_i6` + hspike) | `hmm.predict_i6` + `hmm.hspike.calibrate_i6_emission` | 2 | 3.5 empirical | ✓ | Jaccard 0.979 vs R step17_hmm_i6 (floor 0.96, spec §5.2 target 0.95) |
| `.build_and_add_hspike` | `hmm.hspike.calibrate_i6_emission` (private helpers: `_fit_meanvar_spline`, `_apply_dropout`, …) | 2 | 3.5 empirical | ✓ | measured via the i6 Jaccard row above; stochastic hspike uses `random_state=42` to mirror the R reference script's `set.seed(42)` |
| `get_spike_dists` | `hmm.hspike._get_dists` (private) | 2 | internal | ✓ | covered by `calibrate_i6_emission` public contract |
| `Viterbi.dthmm.adj` kernel (R `inferCNV_HMM.R:1101-1175`) | `kernels.hmm_viterbi_numba._rstyle_log_emit` + `_viterbi_dp_numba` | 2 | kernel-isolated parity | ✓ | Jaccard 0.9999 on R-aligned input (R step15 + step16 → py kernel) per `scripts/triage_i3/triage_real_data.py` |
| `pipeline_phase2.run_phase2` top-level orchestration | `pipeline_phase2.run_phase2` | 2 | — | ✓ | invoked from `pipeline.infercnv()` when `config.HMM=True` |

## Phase 3 (R steps 18-22) — **Implemented**

Status: pipeline_phase3 orchestrator wired into top-level `infercnv()` on
2026-04-25 (post `2026-04-25-phase3-wire-fix-execution` plan). Phase 3
fires whenever `cfg.HMM=True` (R-faithful step 20 per
`inferCNV_ops.R:1463-1499`) or any of `BayesMaxPNormal>0` /
`mask_nonDE_genes` / `denoise` is set.

| R source | Python submodule | Tier target | Status |
|---|---|---|---|
| `inferCNV_BayesNet` (Gibbs, rjags BUGS_Mixture_Model) | `bayesnet/gibbs` | 3.5 \|ΔP\|<0.10 (soft) | Implemented; soft-tier ≥90% on oligo. `reassignCNVs=True` raises (removeCNV-only port). |
| `inferCNV_mask_non_DE` (Wilcoxon/BH per subcluster vs ref) | `mask_de/wilcoxon` | 4 bit-exact | Implemented; `max_diff=1.110e-16`, xor_frac < 1e-4. |
| `clear_noise_via_ref_mean_sd` (`inferCNV_ops.R:2302-2346`) | `denoise/ref_mean_sd` | 4 bit-exact | Implemented; `noise_logistic=True` raises (sigmoidal mask not ported). |
| `assign_HMM_states_to_proxy_expr_vals` (R step 20, i6+i3) | `pipeline_phase3._step20_assign_states_to_proxy_expr_vals` | 4 bit-exact | Implemented; both i6 and i3 `max_diff=0.000e+00`. |
| `filterHighPNormals` (R step 19) | `bayesnet/filter_high_p_normals` | — | Implemented; post-filter Jaccard ≥0.94. |
| `pipeline_phase3.run_phase3` top-level orchestration | `pipeline_phase3.run_phase3` | — | Implemented; persists `bayes_posterior` (canonical key `cnv_posterior`), `de_mask`, `denoised_matrix`, `hmm_proxy_matrix`. |
| `pipeline.infercnv()` ⇒ Phase 3 wire | `pipeline.infercnv()` | — | Implemented; fail-loud guards for `denoise+noise_logistic`, HMM-less Bayes, HMM-less mask. Permanent regression test in `tests/integration/test_top_level_phase3.py`. |

## Phase 3b — **Not implemented**

| R source | Python submodule | Tier target | Status |
|---|---|---|---|
| variational approximation of Gibbs | `bayesnet/vb` | Jaccard≥0.95 vs Gibbs | TODO |

## Skipped (out of scope)

| R source | Rationale |
|---|---|
| `seurat_interaction` | AnnData-only (spec §3.3) |
| `inferCNV_heatmap` (all variants) | Minimal matplotlib port only (`viz/heatmap`); not feature-parity |
| All `save_rds` / RDS checkpointing | AnnData obsm/uns persistence replaces |

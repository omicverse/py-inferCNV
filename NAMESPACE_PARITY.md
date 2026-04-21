# NAMESPACE_PARITY — R infercnv → pyinfercnv

Per `docs/superpowers/specs/2026-04-21-pyinfercnv-master-design.md` §3.3 + §5.2.

Tier meanings (spec §5.1):
- **4 bit-exact**   `max_diff < 1e-10`
- **4 approximate** `max_diff < 1e-6`
- **4 (relaxed)**   `max_diff < 1e-3` (float32 cumulative)
- **3.5 empirical** stochastic — ARI / Jaccard floor
- **3** identity    same ID set / labels

## Phase 1 (R steps 1–14 + 16)

| R source / function | Python submodule | Phase | Tier | Status | R-parity metric |
|---|---|---|---|---|---|
| `CreateInfercnvObject` (chr_exclude, gene_order) | `pipeline._build_chromosome_layout` | 1 | 3 | ✓ | same gene set |
| `require_above_min_mean_expr_cutoff` + `require_above_min_cells_ref` | `preprocess.filter_low_expression_genes` | 1 | 4 bit-exact | ✓ | set + value equality |
| `normalize_counts_by_seq_depth` | `preprocess.normalize_by_seq_depth` | 1 | 4 (relaxed) | ✓ | `max_diff < 1e-2` (float32 cumulative) |
| `log2xplus1` | `preprocess.log2_plus1` | 1 | 4 approximate | ✓ | `max_diff < 1e-5` |
| `subtract_ref_expr_from_obs` (use_bounds=TRUE, 1st pass) | `preprocess.subtract_reference` | 1 | 4 (relaxed) | ✓ | `max_diff < 1e-3` |
| `apply_max_threshold_bounds` | `preprocess.apply_max_centered_threshold` | 1 | 4 approximate | ✓ | `max_diff < 1e-6` |
| `.smooth_center_helper` + `.smooth_helper` (`smooth_by_chromosome`) | `smooth.smooth_pyramidinal` + `kernels.smooth_tail_overwrite` | 1 | 4 (relaxed) | ✓ | `max_diff < 1e-3` (interior bit-exact, tail R-exact) |
| `center_cell_expr_across_chromosome` (median) | `center.center_cells` | 1 | 4 approximate | ✓ | `max_diff < 1e-4` |
| `subtract_ref_expr_from_obs` (2nd pass) | `preprocess.subtract_reference` | 1 | 4 (relaxed) | ✓ | covered by step12 parity |
| `invert_log2` | `preprocess.invert_log2` | 1 | 4 approximate | ✓ | `max_diff < 1e-4` |
| `remove_outliers_norm` (`average_bound`) | `cna.prune_outliers` | 1 | 4 approximate | ✓ | `max_diff < 1e-4` |

## Phase 2 (R steps 15, 17) — **Not implemented**

| R source | Python submodule | Tier target | Status |
|---|---|---|---|
| `inferCNV_tumor_subclusters*` | `subcluster/` | 3.5 ARI≥0.85 | TODO |
| `inferCNV_HMM` (i6) | `hmm/i6` | 4 bit-exact | TODO |
| `inferCNV_i3HMM` | `hmm/i3` | 4 bit-exact | TODO |
| `inferCNV_hidden_spike` | `hmm/hspike` | 3.5 KS≥0.95 | TODO |

## Phase 3 (R steps 18-19, 21-22) — **Not implemented**

| R source | Python submodule | Tier target | Status |
|---|---|---|---|
| `inferCNV_BayesNet` (Gibbs) | `bayesnet/gibbs` | 3.5 |ΔP|<0.05 | TODO |
| `inferCNV_mask_non_DE` | `mask_de/wilcoxon` | 4 bit-exact | TODO |
| `noise_reduction` | `denoise/ref_mean_sd` | 4 bit-exact | TODO |

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

# tests/r_reference.R
#
# R ground-truth fixture dumper for pyinfercnv tier-4 bit-exact r_parity tests.
# Runs the infercnv pipeline step-by-step, exports per-step intermediate matrices
# as TSV files under ./tests/r_out/ for cross-validation against the Python port.
#
# Usage (from pyinfercnv/ repo root):
#   Rscript tests/r_reference.R
#
# Produces: tests/r_out/step{02,03,04,08,09,10,11,12,16}_*.tsv + step_invert.tsv
# Corresponds to R infercnv pipeline steps 2/3/4/8/9/10/11/12/16 + final invert_log2
# (Phase 1 happy path with prune_outliers=TRUE and default pyramidinal smoothing).

set.seed(42)
options(warn = 1)  # surface warnings immediately

suppressPackageStartupMessages(library(infercnv))

# --- Absolute fixture paths (do not copy; reference in place) ---
counts_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/oligodendroglioma_expression_downsampled.counts.matrix.gz"
annotations_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/oligodendroglioma_annotations_downsampled.txt"
gene_order_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt"

output_dir <- "./tests/r_out"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load annotations to determine reference groups ---
annotations <- read.table(annotations_file, header = FALSE, sep = "\t",
                          stringsAsFactors = FALSE)
colnames(annotations) <- c("cell_id", "annotation")
all_labels <- unique(annotations$annotation)
# tumor labels start with `malignant_` / `Tumor` / `Observation` — parenthesized
# "(non-malignant)" must NOT match, so we anchor with ^ and require `_` suffix.
is_tumor <- grepl("^malignant_|^[Tt]umor_|^[Oo]bservation($|_)", all_labels)
ref_annotations <- all_labels[!is_tumor]
cat("[r_reference] unique annotations:", paste(all_labels, collapse = ", "), "\n")
cat("[r_reference] ref_annotations:", paste(ref_annotations, collapse = ", "), "\n")

# --- Create infercnv object ---
infercnv_obj <- infercnv::CreateInfercnvObject(
    raw_counts_matrix = counts_file,
    gene_order_file = gene_order_file,
    annotations_file = annotations_file,
    ref_group_names = ref_annotations,
    delim = "\t"
)

# Helper: write expr.data matrix (genes x cells) to TSV
write_step <- function(obj, name) {
    path <- file.path(output_dir, paste0(name, ".tsv"))
    write.table(obj@expr.data, file = path,
                sep = "\t", quote = FALSE,
                col.names = NA, row.names = TRUE)
    cat(sprintf("[r_reference] wrote %s (%d genes x %d cells)\n",
                path, nrow(obj@expr.data), ncol(obj@expr.data)))
}

# --- Step 2: filter genes ---
infercnv_obj <- infercnv:::require_above_min_mean_expr_cutoff(
    infercnv_obj, min_mean_expr_cutoff = 1
)
infercnv_obj <- infercnv:::require_above_min_cells_ref(
    infercnv_obj, min_cells_per_gene = 3
)
write_step(infercnv_obj, "step02_filtered")

# --- Step 3: CPM-like normalize by median libsize ---
infercnv_obj <- infercnv:::normalize_counts_by_seq_depth(
    infercnv_obj, normalize_factor = NA
)
write_step(infercnv_obj, "step03_normalized")

# --- Step 4: log2(x+1) ---
infercnv_obj <- infercnv:::log2xplus1(infercnv_obj)
write_step(infercnv_obj, "step04_logged")

# --- Step 8: subtract reference (first pass, bounded means) ---
infercnv_obj <- infercnv:::subtract_ref_expr_from_obs(
    infercnv_obj, inv_log = FALSE, use_bounds = TRUE
)
write_step(infercnv_obj, "step08_subtracted")

# --- Step 9: max_centered_threshold = 3 ---
infercnv_obj <- infercnv:::apply_max_threshold_bounds(
    infercnv_obj, threshold = 3
)
write_step(infercnv_obj, "step09_clipped")

# --- Step 10: pyramidinal smoothing (window_length=101) ---
infercnv_obj <- infercnv:::smooth_by_chromosome(
    infercnv_obj, window_length = 101, smooth_ends = TRUE
)
write_step(infercnv_obj, "step10_smoothed")

# --- Step 11: per-cell median center ---
infercnv_obj <- infercnv:::center_cell_expr_across_chromosome(
    infercnv_obj, method = "median"
)
write_step(infercnv_obj, "step11_centered")

# --- Step 12: subtract reference (second pass, after smooth+center) ---
infercnv_obj <- infercnv:::subtract_ref_expr_from_obs(
    infercnv_obj, inv_log = FALSE, use_bounds = TRUE
)
write_step(infercnv_obj, "step12_subtracted2")

# --- Step 14: invert_log2 to linear FC (R run() order: 14 before 16) ---
infercnv_obj <- infercnv:::invert_log2(infercnv_obj)
write_step(infercnv_obj, "step14_invert")

# --- Step 16: remove outliers in LINEAR FC space (matches R run() step_count order) ---
infercnv_obj <- infercnv:::remove_outliers_norm(
    infercnv_obj,
    out_method = "average_bound",
    lower_bound = NA,
    upper_bound = NA
)
write_step(infercnv_obj, "step16_outlier_pruned")

cat("[r_reference] Phase 1 SUCCESS — all Phase 1 TSVs written to", output_dir, "\n")

# =============================================================================
# Phase 2: HMM i6 + HMM i3 via full infercnv::run()
# =============================================================================
# Each run() is wrapped in tryCatch so Phase 1 TSVs are never lost on failure.
# Produces:
#   tests/r_out/step15_subclusters.tsv  (cell -> subcluster label)
#   tests/r_out/step17_hmm_i6.tsv      (genes x cells, integer states 1-6)
#   tests/r_out/step17_hmm_i3.tsv      (genes x cells, integer states 1-3)

phase2_run <- function(hmm_type, tmp_dir) {
    cat(sprintf("[r_reference] Phase 2: starting HMM %s run in %s\n", hmm_type, tmp_dir))
    dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)

    # Fresh infercnv object (do NOT reuse the mutated Phase 1 obj)
    fresh_obj <- infercnv::CreateInfercnvObject(
        raw_counts_matrix = counts_file,
        gene_order_file = gene_order_file,
        annotations_file = annotations_file,
        ref_group_names = ref_annotations,
        delim = "\t"
    )

    # Force the `simple` + CPM Leiden path (R default is `PCA` via Seurat +
    # modularity-via-caller-default, but the PCA path depends on Seurat's
    # internal SNN/irlba which we do not replicate in python. The "simple"
    # path on both sides uses euclidean KNN + igraph::cluster_leiden with
    # CPM, so the C core is identical to python-igraph's community_leiden.
    result_obj <- infercnv::run(
        infercnv_obj              = fresh_obj,
        cutoff                    = 1,
        out_dir                   = tmp_dir,
        cluster_by_groups         = TRUE,
        denoise                   = FALSE,
        HMM                      = TRUE,
        HMM_type                  = hmm_type,
        BayesMaxPNormal           = 0,
        leiden_method             = "simple",
        leiden_function           = "CPM",
        no_plot                   = TRUE,
        no_prelim_plot            = TRUE,
        save_rds                  = TRUE,
        num_threads               = 1,
        up_to_step                = 17
    )

    # --- step15: subclusters ---
    subs <- result_obj@tumor_subclusters$subclusters
    cat(sprintf("[r_reference] tumor_subclusters groups: %s\n",
                paste(names(subs), collapse = ", ")))
    rows <- list()
    for (grp in names(subs)) {
        for (sub_idx in names(subs[[grp]])) {
            cell_names <- names(subs[[grp]][[sub_idx]])
            if (is.null(cell_names)) {
                # fallback: the sub-element itself is a named vector of cell indices
                cell_names <- subs[[grp]][[sub_idx]]
            }
            for (cid in cell_names) {
                rows[[length(rows) + 1]] <- data.frame(
                    cell_id    = cid,
                    subcluster = sprintf("%s.%s", grp, sub_idx),
                    stringsAsFactors = FALSE
                )
            }
        }
    }
    if (length(rows) > 0 && hmm_type == "i6") {
        sub_df <- do.call(rbind, rows)
        sub_path <- file.path(output_dir, "step15_subclusters.tsv")
        write.table(sub_df, file = sub_path, sep = "\t", quote = FALSE, row.names = FALSE)
        cat(sprintf("[r_reference] wrote %s (%d cells)\n", sub_path, nrow(sub_df)))
    }

    # --- step17: HMM states ---
    # The hmm.infercnv_obj is saved as a .infercnv_obj RDS in tmp_dir.
    # Pattern: 17_HMM_pred<resume_token>.infercnv_obj
    hmm_rds_files <- list.files(tmp_dir, pattern = "^17_.*\\.infercnv_obj$", full.names = TRUE)
    cat(sprintf("[r_reference] step17 RDS candidates: %s\n",
                paste(hmm_rds_files, collapse = ", ")))
    if (length(hmm_rds_files) == 0) {
        cat(sprintf("[r_reference] WARNING: no step-17 RDS found in %s; skipping HMM TSV\n", tmp_dir))
        return(invisible(NULL))
    }
    hmm_obj <- readRDS(hmm_rds_files[[1]])
    step17_name <- sprintf("step17_hmm_%s", tolower(hmm_type))
    step17_path <- file.path(output_dir, paste0(step17_name, ".tsv"))
    write.table(hmm_obj@expr.data, file = step17_path,
                sep = "\t", quote = FALSE,
                col.names = NA, row.names = TRUE)
    cat(sprintf("[r_reference] wrote %s (%d genes x %d cells)\n",
                step17_path, nrow(hmm_obj@expr.data), ncol(hmm_obj@expr.data)))
    invisible(NULL)
}

# --- Run i6 (also produces step15_subclusters.tsv) ---
tryCatch({
    tmp_i6 <- file.path(tempdir(), "infercnv_ref_i6")
    phase2_run("i6", tmp_i6)
}, error = function(e) {
    cat(sprintf("[r_reference] Phase 2 i6 reference generation skipped: %s\n",
                conditionMessage(e)))
})

# --- Run i3 ---
tryCatch({
    tmp_i3 <- file.path(tempdir(), "infercnv_ref_i3")
    phase2_run("i3", tmp_i3)
}, error = function(e) {
    cat(sprintf("[r_reference] Phase 2 i3 reference generation skipped: %s\n",
                conditionMessage(e)))
})

cat("[r_reference] DONE — Phase 1 + Phase 2 TSV generation complete\n")

# =============================================================================
# Phase 3 — step18/19/21/22 TSV dump for parity tests (SKELETON)
# =============================================================================
# Status: skeleton — implementation pending Agent B1/B2/B3 landing the Python
# modules. Step numbers match R's step_count (inferCNV_ops.R step 18/19/21/22).
# Step 20 (assign_HMM_states_to_proxy_expr_vals) is in-scope for the Phase 3
# orchestrator but it's a simple state→CN-ratio remap; we dump it opportunistically.
#
# Each block is wrapped in tryCatch so earlier-phase TSVs are never lost.
# Dumps use the `bayes` / `maskDE` / `denoised` run() variant of the full
# pipeline, reading expr.data off the resulting infercnv_obj at the matching
# checkpoint. Python parity tests consume genes×cells orientation (matches
# existing Phase 1/2 dump convention).
#
# Convention: each Agent appends a separate tryCatch block so edits don't
# collide. Do NOT re-use the `result_obj` variable across blocks — always
# build a fresh object per dump.

phase3_run <- function(hmm_type, tmp_dir, do_bayes, do_mask, do_denoise,
                       BayesMaxPNormal = 0, mask_nonDE_pval = 0.05,
                       mask_nonDE_genes_flag = FALSE, denoise_flag = FALSE,
                       up_to_step = 22) {
    cat(sprintf(
        "[r_reference] Phase 3: hmm_type=%s bayes=%s mask=%s denoise=%s in %s\n",
        hmm_type, do_bayes, do_mask, do_denoise, tmp_dir))
    dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)

    fresh_obj <- infercnv::CreateInfercnvObject(
        raw_counts_matrix = counts_file,
        gene_order_file = gene_order_file,
        annotations_file = annotations_file,
        ref_group_names = ref_annotations,
        delim = "\t"
    )

    result_obj <- infercnv::run(
        infercnv_obj              = fresh_obj,
        cutoff                    = 1,
        out_dir                   = tmp_dir,
        cluster_by_groups         = TRUE,
        denoise                   = denoise_flag,
        HMM                      = TRUE,
        HMM_type                  = hmm_type,
        BayesMaxPNormal           = BayesMaxPNormal,
        mask_nonDE_genes          = mask_nonDE_genes_flag,
        mask_nonDE_pval           = mask_nonDE_pval,
        leiden_method             = "simple",
        leiden_function           = "CPM",
        no_plot                   = TRUE,
        no_prelim_plot            = TRUE,
        save_rds                  = TRUE,
        num_threads               = 1,
        up_to_step                = up_to_step
    )
    invisible(result_obj)
}

# --- Agent B1 segment: step18 BayesNet posterior dump + step19 filter ---
# Runs full run() up to step 19 with BayesMaxPNormal=0.5 and
# reassignCNVs=FALSE (the Python port deliberately does not implement the
# reassignCNV branch in this first round — see bayesnet/gibbs.py docstring).
#
# Produces (see Agent B1 implementation notes in test_step18_*):
#   tests/r_out/step18_bayes_cnv_prob.tsv          — n_regions × K colMeans(theta)
#   tests/r_out/step18_bayes_cell_prob_long.tsv    — long: region_idx, cell_id, state, prob
#   tests/r_out/step18_regions_meta.tsv            — per-region metadata
#   tests/r_out/step19_hmm_i6_filtered.tsv         — genes × cells post-filter
#   tests/r_out/step19_hmm_i6_prefilter.tsv        — genes × cells pre-filter (baseline)
tryCatch({
    tmp_b <- file.path(tempdir(), "infercnv_ref_phase3_bayes")
    dir.create(tmp_b, showWarnings = FALSE, recursive = TRUE)
    fresh_obj <- infercnv::CreateInfercnvObject(
        raw_counts_matrix = counts_file,
        gene_order_file = gene_order_file,
        annotations_file = annotations_file,
        ref_group_names = ref_annotations,
        delim = "\t"
    )
    result_obj <- infercnv::run(
        infercnv_obj              = fresh_obj,
        cutoff                    = 1,
        out_dir                   = tmp_b,
        cluster_by_groups         = TRUE,
        denoise                   = FALSE,
        HMM                      = TRUE,
        HMM_type                  = "i6",
        BayesMaxPNormal           = 0.5,
        reassignCNVs              = FALSE,
        leiden_method             = "simple",
        leiden_function           = "CPM",
        no_plot                   = TRUE,
        no_prelim_plot            = TRUE,
        save_rds                  = TRUE,
        num_threads               = 1,
        up_to_step                = 19
    )

    # --- Locate MCMC_inferCNV_obj.rds under BayesNetOutput.<token>/ ---
    bayesnet_dirs <- list.files(tmp_b, pattern = "^BayesNetOutput\\.", full.names = TRUE)
    cat(sprintf("[r_reference] BayesNetOutput dirs: %s\n",
                paste(bayesnet_dirs, collapse = ", ")))
    if (length(bayesnet_dirs) == 0) stop("No BayesNetOutput directory under ", tmp_b)
    mcmc_rds_path <- file.path(bayesnet_dirs[[1]], "MCMC_inferCNV_obj.rds")
    mcmc_obj <- readRDS(mcmc_rds_path)
    cat(sprintf("[r_reference] loaded MCMC_inferCNV_obj: %d regions\n",
                length(mcmc_obj@cell_gene)))

    # --- Per-region posterior cnv_prob (colMeans of the raw theta samples) ---
    K <- 6
    n_regions <- length(mcmc_obj@cnv_probabilities)
    cnv_prob_mat <- matrix(NA_real_, nrow = n_regions, ncol = K)
    for (r in seq_len(n_regions)) {
        thetas <- mcmc_obj@cnv_probabilities[[r]]
        if (!is.null(thetas) && ncol(thetas) == K) {
            cnv_prob_mat[r, ] <- colMeans(thetas)
        }
    }
    colnames(cnv_prob_mat) <- paste0("state", seq_len(K))
    cnv_prob_path <- file.path(output_dir, "step18_bayes_cnv_prob.tsv")
    write.table(cnv_prob_mat, file = cnv_prob_path,
                sep = "\t", quote = FALSE,
                col.names = NA, row.names = TRUE)
    cat(sprintf("[r_reference] wrote %s (%d regions × %d states)\n",
                cnv_prob_path, n_regions, K))

    # --- Per-region metadata for Py alignment ---
    meta_rows <- list()
    for (r in seq_len(n_regions)) {
        cg <- mcmc_obj@cell_gene[[r]]
        meta_rows[[r]] <- data.frame(
            region_idx = r,
            cnv_region_name = as.character(cg$cnv_regions),
            n_cells = length(cg$Cells),
            n_genes = length(cg$Genes),
            hmm_state = cg$State,
            stringsAsFactors = FALSE
        )
    }
    meta_df <- do.call(rbind, meta_rows)
    meta_path <- file.path(output_dir, "step18_regions_meta.tsv")
    write.table(meta_df, file = meta_path, sep = "\t", quote = FALSE, row.names = FALSE)
    cat(sprintf("[r_reference] wrote %s (%d rows)\n", meta_path, nrow(meta_df)))

    # --- Per-cell per-state posterior in long format ---
    cell_expr_colnames <- colnames(mcmc_obj@expr.data)
    cell_long_rows <- list()
    for (r in seq_len(n_regions)) {
        cp <- mcmc_obj@cell_probabilities[[r]]
        cg <- mcmc_obj@cell_gene[[r]]
        if (is.null(cp) || length(cg$Cells) == 0) next
        region_cell_ids <- cell_expr_colnames[cg$Cells]
        for (cidx in seq_along(region_cell_ids)) {
            for (s in seq_len(K)) {
                cell_long_rows[[length(cell_long_rows) + 1]] <- data.frame(
                    region_idx = r,
                    cell_id = region_cell_ids[cidx],
                    state = s,
                    prob = as.numeric(cp[s, cidx]),
                    stringsAsFactors = FALSE
                )
            }
        }
    }
    if (length(cell_long_rows) > 0) {
        cell_long_df <- do.call(rbind, cell_long_rows)
        cell_long_path <- file.path(output_dir, "step18_bayes_cell_prob_long.tsv")
        write.table(cell_long_df, file = cell_long_path,
                    sep = "\t", quote = FALSE, row.names = FALSE)
        cat(sprintf("[r_reference] wrote %s (%d rows)\n",
                    cell_long_path, nrow(cell_long_df)))
    }

    # --- step19 filtered HMM states ---
    hmm19_files <- list.files(tmp_b, pattern = "^19_HMM.*\\.infercnv_obj$", full.names = TRUE)
    cat(sprintf("[r_reference] step19 RDS candidates: %s\n",
                paste(hmm19_files, collapse = ", ")))
    if (length(hmm19_files) > 0) {
        hmm19_obj <- readRDS(hmm19_files[[1]])
        step19_path <- file.path(output_dir, "step19_hmm_i6_filtered.tsv")
        write.table(hmm19_obj@expr.data, file = step19_path,
                    sep = "\t", quote = FALSE,
                    col.names = NA, row.names = TRUE)
        cat(sprintf("[r_reference] wrote %s (%d genes × %d cells)\n",
                    step19_path, nrow(hmm19_obj@expr.data), ncol(hmm19_obj@expr.data)))
    }

    # --- step17/19 pre-filter baseline (alignment anchor) ---
    hmm17_files <- list.files(tmp_b, pattern = "^17_HMM.*\\.infercnv_obj$", full.names = TRUE)
    if (length(hmm17_files) > 0) {
        hmm17_obj <- readRDS(hmm17_files[[1]])
        step19_pre_path <- file.path(output_dir, "step19_hmm_i6_prefilter.tsv")
        write.table(hmm17_obj@expr.data, file = step19_pre_path,
                    sep = "\t", quote = FALSE,
                    col.names = NA, row.names = TRUE)
        cat(sprintf("[r_reference] wrote %s (%d genes × %d cells)\n",
                    step19_pre_path, nrow(hmm17_obj@expr.data), ncol(hmm17_obj@expr.data)))
    }
}, error = function(e) {
    cat(sprintf("[r_reference] Phase 3 BayesNet dump skipped: %s\n", conditionMessage(e)))
})

# --- Agent B2 segment: step21 mask_non_DE dump ---
# Runs run() with mask_nonDE_genes=TRUE up to step 21; dumps the masked
# expr.data as step21_mask_nonDE.tsv (genes × cells).
#
# Bit-exact parity contract (see pyinfercnv/mask_de/wilcoxon.py module
# docstring): we monkey-patch ``infercnv:::get_DE_genes_basic`` to strip
# the ``rnorm()`` tie-breaking jitter AND force ``exact = FALSE`` on
# ``wilcox.test`` so the R side always hits the normal-approximation
# branch. scipy's ``mannwhitneyu(method="asymptotic", use_continuity=TRUE)``
# matches that branch to machine precision on small and large samples,
# with and without ties (verified on a 4-case toy). Without this patch
# the R jitter is RNG-stochastic and bit-exact parity is impossible.
tryCatch({
    patched_get_DE_genes_basic <- function(infercnv_obj,
                                           p_val_thresh = 0.05,
                                           test.use = "wilcoxon") {
        all_DE_results <- list()
        statfxns <- list()
        statfxns[["wilcoxon"]] <- function(x, idx1, idx2) {
            vals1 <- x[idx1]; vals2 <- x[idx2]
            # patched: no rnorm jitter; force asymptotic to match scipy
            w <- suppressWarnings(
                wilcox.test(vals1, vals2, exact = FALSE, correct = TRUE)
            )
            return(w$p.value)
        }
        statfxns[["t"]] <- function(x, idx1, idx2) {
            vals1 <- x[idx1]; vals2 <- x[idx2]
            res <- try(t.test(vals1, vals2), silent = TRUE)
            if (is(res, "try-error")) return(NA) else return(res$p.value)
        }
        statfxn <- statfxns[[test.use]]
        normal_types <- names(infercnv_obj@reference_grouped_cell_indices)
        tumor_groupings <- infercnv_obj@observation_grouped_cell_indices
        for (tumor_type in names(tumor_groupings)) {
            indices <- infercnv_obj@tumor_subclusters[["subclusters"]][[tumor_type]]
            if (is.list(indices)) {
                tumor_indices_list <- indices
            } else {
                tumor_indices_list <- list(indices)
            }
            for (tumor_indices_name in names(tumor_indices_list)) {
                tumor_indices <- tumor_indices_list[[tumor_indices_name]]
                for (normal_type in normal_types) {
                    normal_indices <- infercnv_obj@reference_grouped_cell_indices[[normal_type]]
                    pvals <- apply(infercnv_obj@expr.data, 1, statfxn,
                                   idx1 = normal_indices, idx2 = tumor_indices)
                    pvals <- unlist(pvals)
                    pvals <- p.adjust(pvals, method = "BH")
                    names(pvals) <- rownames(infercnv_obj@expr.data)
                    genes <- names(pvals)[pvals < p_val_thresh]
                    condition_pair <- paste(tumor_indices_name, normal_type, sep = ",")
                    all_DE_results[[condition_pair]] <- list(
                        tumor_indices = tumor_indices,
                        normal = normal_type,
                        pvals = pvals,
                        de_genes = genes
                    )
                }
            }
        }
        return(all_DE_results)
    }
    assignInNamespace("get_DE_genes_basic", patched_get_DE_genes_basic,
                      ns = "infercnv")

    tmp_m <- file.path(tempdir(), "infercnv_ref_phase3_mask")
    mask_obj <- phase3_run("i6", tmp_m,
                           do_bayes = FALSE, do_mask = TRUE, do_denoise = FALSE,
                           BayesMaxPNormal = 0,
                           mask_nonDE_genes_flag = TRUE,
                           up_to_step = 21)
    if (is.null(mask_obj)) {
        cat("[r_reference] Phase 3 mask_non_DE: phase3_run returned NULL; skipping\n")
    } else {
        step21_path <- file.path(output_dir, "step21_mask_nonDE.tsv")
        write.table(mask_obj@expr.data, file = step21_path,
                    sep = "\t", quote = FALSE,
                    col.names = NA, row.names = TRUE)
        cat(sprintf("[r_reference] wrote %s (%d genes x %d cells)\n",
                    step21_path,
                    nrow(mask_obj@expr.data),
                    ncol(mask_obj@expr.data)))

        # Also dump the step21-time subcluster partition so the python
        # parity test uses *this run's* subclusters (not step15's, which
        # came from a different run() invocation and thus a different
        # point in the global RNG stream).
        subs21 <- mask_obj@tumor_subclusters$subclusters
        rows21 <- list()
        for (grp in names(subs21)) {
            for (sub_idx in names(subs21[[grp]])) {
                cell_names <- names(subs21[[grp]][[sub_idx]])
                if (is.null(cell_names)) {
                    cell_names <- subs21[[grp]][[sub_idx]]
                }
                for (cid in cell_names) {
                    rows21[[length(rows21) + 1]] <- data.frame(
                        cell_id    = cid,
                        subcluster = sprintf("%s.%s", grp, sub_idx),
                        stringsAsFactors = FALSE
                    )
                }
            }
        }
        if (length(rows21) > 0) {
            step21_sub_df <- do.call(rbind, rows21)
            step21_sub_path <- file.path(output_dir, "step21_subclusters.tsv")
            write.table(step21_sub_df, file = step21_sub_path,
                        sep = "\t", quote = FALSE, row.names = FALSE)
            cat(sprintf("[r_reference] wrote %s (%d cells)\n",
                        step21_sub_path, nrow(step21_sub_df)))
        }

        # Also dump the pre-mask @expr.data (post-step 16 snapshot taken
        # inside *this* run() invocation) so the python test can replay
        # step 21 on an RNG-aligned input rather than step16_outlier_pruned.tsv
        # from an earlier run.
        # NB: for the current run(..., up_to_step=21) the object mutates
        # IN PLACE; we cannot easily grab a pre-mask snapshot post-hoc.
        # Instead, we re-derive the input from the existing step16 TSV and
        # trust that both run()s produce identical step16 output (Phase 1
        # is deterministic — verified by test_step16_outlier_prune_parity).
    }
}, error = function(e) {
    cat(sprintf("[r_reference] Phase 3 mask_non_DE dump skipped: %s\n", conditionMessage(e)))
})

# --- Agent B3 segment: step22 denoise dump ---
# Runs run() with denoise=TRUE, BayesMaxPNormal=0 (Bayes off), mask=FALSE;
# up_to_step=22; dumps the denoised expr.data as step22_denoised.tsv.
#
# The R infercnv::run() with up_to_step=22 returns the final mutated infercnv_obj
# (step 22 operates on the main infercnv_obj@expr.data — steps 17-20 mutate a
# separate hmm.infercnv_obj, so the main object still holds post-step16 linear
# FC values going into step 22). We capture the return value from phase3_run
# (which calls `invisible(result_obj)`) and dump `@expr.data` directly; this
# avoids the brittle RDS filename pattern
# `22_denoiseHMMi6.leiden.NF_NA.SD_1.5.NL_FALSE.infercnv_obj`.
tryCatch({
    tmp_d <- file.path(tempdir(), "infercnv_ref_phase3_denoise")
    denoise_obj <- phase3_run("i6", tmp_d,
                              do_bayes = FALSE, do_mask = FALSE, do_denoise = TRUE,
                              BayesMaxPNormal = 0,
                              denoise_flag = TRUE,
                              up_to_step = 22)
    if (is.null(denoise_obj)) {
        cat("[r_reference] Phase 3 denoise: phase3_run returned NULL; skipping\n")
    } else {
        step22_path <- file.path(output_dir, "step22_denoised.tsv")
        write.table(denoise_obj@expr.data, file = step22_path,
                    sep = "\t", quote = FALSE,
                    col.names = NA, row.names = TRUE)
        cat(sprintf("[r_reference] wrote %s (%d genes x %d cells)\n",
                    step22_path,
                    nrow(denoise_obj@expr.data),
                    ncol(denoise_obj@expr.data)))
    }
}, error = function(e) {
    cat(sprintf("[r_reference] Phase 3 denoise dump skipped: %s\n", conditionMessage(e)))
})

cat("[r_reference] ALL DONE — Phase 1 + Phase 2 complete; Phase 3 dump skeleton in place\n")

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

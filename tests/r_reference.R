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

# --- Step 16: remove outliers (average_bound method) ---
infercnv_obj <- infercnv:::remove_outliers_norm(
    infercnv_obj,
    out_method = "average_bound",
    lower_bound = NA,
    upper_bound = NA
)
write_step(infercnv_obj, "step16_outlier_pruned")

# --- Final: invert_log2 back to linear FC ---
infercnv_obj <- infercnv:::invert_log2(infercnv_obj)
write_step(infercnv_obj, "step_invert")

cat("[r_reference] SUCCESS — all TSVs written to", output_dir, "\n")

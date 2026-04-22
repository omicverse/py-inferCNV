# examples/r_driver_phase2.R
#
# Side-by-side R reference for tutorial_phase2.ipynb.
#
# Runs the upstream Broad Institute `infercnv::run` Phase 2 pipeline twice
# (HMM_type="i6" then HMM_type="i3") on the oligodendroglioma downsampled
# fixture. Matches the r_reference.R settings that pyinfercnv's tier-3.5
# R-parity tests (`tests/test_r_parity.py`) use to assert Jaccard floors.
#
# Critical: leiden_method="simple" + leiden_function="CPM" force the R
# Leiden code path whose C core matches Python's python-igraph. The default
# PCA+Seurat path has no Python equivalent and causes subcluster ARI
# divergence — see `tests/r_reference.R` lines 143-148 and the
# https://github.com/broadinstitute/inferCNV/wiki/infercnv-tumor-subclusters
# wiki page on leiden_method.
#
# Usage (from pyinfercnv/ repo root):
#
#   Rscript examples/r_driver_phase2.R
#
# Writes outputs under examples/r_out_phase2/:
#   - infercnv_run_i6/   step 17 RDS checkpoints, HMM_type=i6
#   - infercnv_run_i3/   step 17 RDS checkpoints, HMM_type=i3
#   - r_phase2_summary.txt (elapsed per HMM type, cells/genes)

suppressPackageStartupMessages(library(infercnv))
options(scipen = 100)  # infercnv itself warns to set this before subclusters mode

counts_file      <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/oligodendroglioma_expression_downsampled.counts.matrix.gz"
annotations_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/oligodendroglioma_annotations_downsampled.txt"
gene_order_file  <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt"

out_dir <- file.path("examples", "r_out_phase2")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

annotations <- read.table(
  annotations_file, header = FALSE, sep = "\t", stringsAsFactors = FALSE
)
colnames(annotations) <- c("cell_id", "annotation")
all_labels <- unique(annotations$annotation)
is_tumor <- grepl("^malignant_|^[Tt]umor_|^[Oo]bservation($|_)", all_labels)
ref_annotations <- all_labels[!is_tumor]
cat("[r_driver_phase2] ref groups:", paste(ref_annotations, collapse = ", "), "\n")

run_hmm <- function(hmm_type) {
  work_dir <- file.path(out_dir, paste0("infercnv_run_", hmm_type))
  dir.create(work_dir, showWarnings = FALSE, recursive = TRUE)

  # Fresh object per call — infercnv::run mutates in place, so reusing would
  # carry over the previous HMM_type's state.
  obj <- infercnv::CreateInfercnvObject(
    raw_counts_matrix = counts_file,
    gene_order_file   = gene_order_file,
    annotations_file  = annotations_file,
    ref_group_names   = ref_annotations,
    delim             = "\t"
  )
  n_cells <- ncol(obj@expr.data)
  n_genes <- nrow(obj@expr.data)
  cat(sprintf("[r_driver_phase2] %s: starting on %d cells x %d genes\n",
              hmm_type, n_cells, n_genes))

  t0 <- Sys.time()
  obj <- infercnv::run(
    obj,
    cutoff            = 1,            # smart-seq2 cutoff; 10x data uses 0.1
    out_dir           = work_dir,
    cluster_by_groups = TRUE,
    denoise           = FALSE,         # Phase 3 — keep off for parity runs
    HMM               = TRUE,
    HMM_type          = hmm_type,
    BayesMaxPNormal   = 0,
    leiden_method     = "simple",      # python-igraph-compatible leiden path
    leiden_function   = "CPM",
    up_to_step        = 17,            # HMM states, no BayesNet / denoise
    num_threads       = 1,             # deterministic ordering
    no_plot           = TRUE,
    no_prelim_plot    = TRUE,
    save_rds          = TRUE
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("[r_driver_phase2] %s elapsed %.2f s\n", hmm_type, elapsed))
  list(hmm_type = hmm_type, elapsed = elapsed, n_cells = n_cells, n_genes = n_genes)
}

results <- lapply(c("i6", "i3"), run_hmm)

# --- Summary --------------------------------------------------------------
summary_path <- file.path(out_dir, "r_phase2_summary.txt")
lines <- c(
  sprintf("fixture = oligodendroglioma_expression_downsampled"),
  sprintf("cutoff = 1  (smart-seq2 convention; 10x uses 0.1)"),
  sprintf("leiden_method = simple"),
  sprintf("leiden_function = CPM"),
  sprintf("BayesMaxPNormal = 0   (Phase 3 off)"),
  sprintf("denoise = FALSE")
)
for (r in results) {
  lines <- c(
    lines,
    sprintf("HMM_type=%s   elapsed_s=%.3f   n_cells=%d   n_genes=%d",
            r$hmm_type, r$elapsed, r$n_cells, r$n_genes)
  )
}
writeLines(lines, con = summary_path)
cat(sprintf("[r_driver_phase2] wrote %s\n", summary_path))

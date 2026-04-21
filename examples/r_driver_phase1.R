# examples/r_driver_phase1.R
#
# Side-by-side R reference for tutorial_phase1.ipynb.
#
# Runs the upstream Broad Institute `infercnv::run` pipeline through step 14
# (log2 -> linear FC) on the same oligodendroglioma downsampled fixture the
# Python tutorial uses — Microglia + Oligodendrocytes as the reference group.
#
# Usage (from pyinfercnv/ repo root):
#
#   Rscript examples/r_driver_phase1.R
#
# Writes outputs under examples/r_out_phase1/:
#   - the infercnv::run working directory (figures, HMM-free Phase 1 artefacts)
#   - r_phase1_summary.txt (cells / genes / runtime)
#
# This script is intentionally minimal — it stops at up_to_step = 14 so it
# mirrors the Python Phase 1 scope (no HMM / no BayesNet).

suppressPackageStartupMessages(library(infercnv))

# --- Absolute fixture paths (shared with tests/r_reference.R) -------------
counts_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/oligodendroglioma_expression_downsampled.counts.matrix.gz"
annotations_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/oligodendroglioma_annotations_downsampled.txt"
gene_order_file <- "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt"

out_dir <- file.path("examples", "r_out_phase1")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Reference groups: every non-`malignant_*` annotation ------------------
annotations <- read.table(
  annotations_file, header = FALSE, sep = "\t", stringsAsFactors = FALSE
)
colnames(annotations) <- c("cell_id", "annotation")
all_labels <- unique(annotations$annotation)
is_tumor <- grepl("^malignant_|^[Tt]umor_|^[Oo]bservation($|_)", all_labels)
ref_annotations <- all_labels[!is_tumor]

cat("[r_driver_phase1] annotations:", paste(all_labels, collapse = ", "), "\n")
cat("[r_driver_phase1] reference  :", paste(ref_annotations, collapse = ", "), "\n")

# --- Build the infercnv object --------------------------------------------
infercnv_obj <- infercnv::CreateInfercnvObject(
  raw_counts_matrix = counts_file,
  gene_order_file   = gene_order_file,
  annotations_file  = annotations_file,
  ref_group_names   = ref_annotations,
  delim             = "\t"
)

n_cells <- ncol(infercnv_obj@expr.data)
n_genes <- nrow(infercnv_obj@expr.data)
cat(sprintf("[r_driver_phase1] loaded %d cells x %d genes\n", n_cells, n_genes))

# --- Run Phase 1 only (up_to_step = 14) -----------------------------------
# Matches pyinfercnv's Phase 1 scope: step 16 (outlier prune) is part of the
# default path in the Python pipeline, but up_to_step = 14 keeps the R run
# small and deterministic; flip to 17 if you want the pruned comparison.
t0 <- Sys.time()
infercnv_obj <- infercnv::run(
  infercnv_obj,
  cutoff              = 0.1,
  out_dir             = out_dir,
  cluster_by_groups   = TRUE,
  denoise             = FALSE,
  HMM                 = FALSE,
  analysis_mode       = "samples",
  up_to_step          = 14,
  num_threads         = 4,
  no_plot             = FALSE,
  write_expr_matrix   = TRUE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[r_driver_phase1] infercnv::run up_to_step=14 elapsed %.2f s\n", elapsed))

# --- Small summary file (parallel to Python tutorial's profile dict) ------
summary_path <- file.path(out_dir, "r_phase1_summary.txt")
writeLines(
  c(
    sprintf("n_cells   = %d", n_cells),
    sprintf("n_genes   = %d", n_genes),
    sprintf("elapsed_s = %.3f", elapsed),
    sprintf("ref_group = %s", paste(ref_annotations, collapse = " | ")),
    sprintf("up_to_step = 14")
  ),
  con = summary_path
)
cat(sprintf("[r_driver_phase1] wrote %s\n", summary_path))

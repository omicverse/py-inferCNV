# scripts/phase2_benchmark/run_r_phase2.R
#
# Run R infercnv::run() up_to_step=17 on a single patient + HMM_type.
# Writes step15 subcluster groupings + step17 HMM state matrix as TSV alongside
# a timing summary, so compare_py_vs_r.py can diff them vs pyinfercnv outputs.
#
# Usage (from pyinfercnv/ repo root):
#   Rscript scripts/phase2_benchmark/run_r_phase2.R \
#       --counts benchmarks/phase2/<Cancer>/<Patient>/counts.tsv \
#       --annotations benchmarks/phase2/<Cancer>/<Patient>/annotations_phase2.txt \
#       --gene-order benchmarks/phase2/gene_order_hg38.tsv \
#       --out-dir benchmarks/phase2/<Cancer>/<Patient>/r_out \
#       --hmm-type i6 \
#       --num-threads 4
#
# Resumable: skips if <out_dir>/r_<hmm_type>_summary.txt exists and HMM RDS is
# present. Delete the summary file to force a re-run.

suppressPackageStartupMessages(library(infercnv))
# infercnv itself warns to set this before analysis_mode="subclusters"; without
# it the internal hclust call can choke on scientific-notation ambiguity.
options(scipen = 100)

# Manual --flag value parser (avoids the optparse dependency).
parse_cli <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    out <- list(
        counts = NULL, annotations = NULL, `gene-order` = NULL,
        `out-dir` = NULL, `hmm-type` = "i6",
        `num-threads` = 4L, cutoff = 0.1
    )
    i <- 1
    while (i <= length(args)) {
        key <- sub("^--", "", args[i])
        if (!key %in% names(out)) {
            stop(sprintf("unknown flag: %s", args[i]))
        }
        val <- args[i + 1L]
        if (key %in% c("num-threads")) val <- as.integer(val)
        else if (key == "cutoff") val <- as.double(val)
        out[[key]] <- val
        i <- i + 2L
    }
    out
}
opt <- parse_cli()

stopifnot(!is.null(opt$counts), !is.null(opt$annotations),
          !is.null(opt$`gene-order`), !is.null(opt$`out-dir`))
stopifnot(opt$`hmm-type` %in% c("i6", "i3"))

hmm_type <- opt$`hmm-type`
out_dir  <- opt$`out-dir`
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
work_dir <- file.path(out_dir, paste0("infercnv_run_", hmm_type))
dir.create(work_dir, showWarnings = FALSE, recursive = TRUE)
summary_path <- file.path(out_dir, sprintf("r_%s_summary.txt", hmm_type))
state_path   <- file.path(out_dir, sprintf("step17_hmm_%s.tsv", hmm_type))
group_path   <- file.path(out_dir, sprintf("step15_subclusters_%s.tsv", hmm_type))

# --- Resumability ---------------------------------------------------------
if (file.exists(summary_path) && file.exists(state_path) && file.exists(group_path)) {
    cat(sprintf("[run_r_phase2] skip (already done): %s\n", summary_path))
    quit(save = "no", status = 0)
}

# --- Parse reference group labels from annotations file -------------------
annotations <- read.table(
    opt$annotations, header = FALSE, sep = "\t", stringsAsFactors = FALSE
)
colnames(annotations) <- c("cell_id", "annotation")
all_labels <- unique(annotations$annotation)
is_tumor   <- grepl("^malignant_", all_labels)
ref_labels <- all_labels[!is_tumor]
cat(sprintf("[run_r_phase2] ref groups: %s\n", paste(ref_labels, collapse = ", ")))

# --- Build object ---------------------------------------------------------
obj <- infercnv::CreateInfercnvObject(
    raw_counts_matrix = opt$counts,
    gene_order_file   = opt$`gene-order`,
    annotations_file  = opt$annotations,
    ref_group_names   = ref_labels,
    delim             = "\t"
)
n_cells <- ncol(obj@expr.data)
n_genes <- nrow(obj@expr.data)
cat(sprintf("[run_r_phase2] post-Create: %d genes x %d cells\n", n_genes, n_cells))

# --- Run Phase 2 (up_to_step=17, HMM on) ---------------------------------
# Settings mirror tests/r_reference.R (the fixture that achieves i3=0.976 /
# i6=0.968 Jaccard on oligodendroglioma). Critical: leiden_method="simple" +
# leiden_function="CPM" force a code path whose C core matches python-igraph's
# community_leiden; the R default PCA-via-Seurat path has no Python analogue
# and causes subcluster ARI collapse → HMM input divergence → low Jaccard.
t0 <- Sys.time()
obj <- infercnv::run(
    obj,
    cutoff                   = opt$cutoff,
    out_dir                  = work_dir,
    cluster_by_groups        = TRUE,
    denoise                  = FALSE,
    HMM                      = TRUE,
    HMM_type                 = hmm_type,
    BayesMaxPNormal          = 0,
    leiden_method            = "simple",
    leiden_function          = "CPM",
    up_to_step               = 17,
    num_threads              = opt$`num-threads`,
    no_plot                  = TRUE,
    no_prelim_plot           = TRUE,
    save_rds                 = TRUE,
    write_expr_matrix        = FALSE,
    resume_mode              = TRUE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[run_r_phase2] HMM=%s elapsed %.2f s\n", hmm_type, elapsed))

# --- Dump step17 HMM state matrix -----------------------------------------
# infercnv writes 17_HMM_pred<resume_token>.infercnv_obj into work_dir; the
# object's @expr.data is the integer state matrix genes x cells.
rds_candidates <- sort(list.files(
    work_dir, pattern = "^17_HMM_pred.*\\.infercnv_obj$", full.names = TRUE
))
if (length(rds_candidates) == 0) {
    stop(sprintf("[run_r_phase2] no step-17 RDS found in %s", work_dir))
}
hmm_obj <- readRDS(rds_candidates[1])
write.table(
    hmm_obj@expr.data, file = state_path, sep = "\t", quote = FALSE,
    col.names = NA, row.names = TRUE
)
cat(sprintf("[run_r_phase2] wrote %s (%d x %d)\n",
            state_path, nrow(hmm_obj@expr.data), ncol(hmm_obj@expr.data)))

# --- Dump step15 subcluster grouping (cell_id, subcluster_label) ----------
# hmm_obj@tumor_subclusters$subclusters is a list keyed by group (annotation)
# with sublists keyed by subcluster id and values = cell indices into expr.data.
# Flatten to a (cell_id, subcluster_label) TSV.
subc <- hmm_obj@tumor_subclusters$subclusters
cell_names <- colnames(hmm_obj@expr.data)
rows <- list()
for (group in names(subc)) {
    for (subid in names(subc[[group]])) {
        idx <- subc[[group]][[subid]]
        for (ix in idx) {
            rows[[length(rows) + 1L]] <- data.frame(
                cell_id = cell_names[ix],
                subcluster = sprintf("%s.%s", group, subid),
                stringsAsFactors = FALSE
            )
        }
    }
}
if (length(rows) == 0) {
    stop("[run_r_phase2] no subclusters extracted from hmm_obj")
}
subc_df <- do.call(rbind, rows)
write.table(subc_df, file = group_path, sep = "\t", quote = FALSE,
            col.names = TRUE, row.names = FALSE)
cat(sprintf("[run_r_phase2] wrote %s (%d rows)\n", group_path, nrow(subc_df)))

# --- Summary --------------------------------------------------------------
writeLines(c(
    sprintf("hmm_type  = %s", hmm_type),
    sprintf("n_cells   = %d", n_cells),
    sprintf("n_genes   = %d", n_genes),
    sprintf("elapsed_s = %.3f", elapsed),
    sprintf("ref_groups = %s", paste(ref_labels, collapse = " | ")),
    sprintf("up_to_step = 17"),
    sprintf("cutoff = %.3f", opt$cutoff),
    sprintf("num_threads = %d", opt$`num-threads`)
), con = summary_path)
cat(sprintf("[run_r_phase2] wrote %s\n", summary_path))

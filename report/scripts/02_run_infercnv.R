# R inferCNV on one task. Usage: Rscript 02_run_infercnv.R <task_dir> <gene_order_file> <out_dir> [seed] [cutoff]
# Settings: cluster_by_groups, denoise, HMM (i6), no BayesNet filter, one thread, set.seed(seed).
suppressPackageStartupMessages({library(infercnv); library(Matrix); library(jsonlite)}); options(scipen = 100)
a <- commandArgs(TRUE); td <- a[1]; gof <- a[2]; od <- a[3]
seed <- if (length(a) > 3) as.integer(a[4]) else 1L; cutoff <- if (length(a) > 4) as.numeric(a[5]) else 0.1
# per-task overrides: cutoff.txt, gene_order.tsv, ref_groups.txt (default reference group "normal")
if (file.exists(file.path(td, "cutoff.txt"))) cutoff <- as.numeric(readLines(file.path(td, "cutoff.txt"))[1])
if (file.exists(file.path(td, "gene_order.tsv"))) gof <- file.path(td, "gene_order.tsv")
refs <- if (file.exists(file.path(td, "ref_groups.txt"))) readLines(file.path(td, "ref_groups.txt")) else "normal"
dir.create(od, recursive = TRUE, showWarnings = FALSE)
m <- as(readMM(file.path(td, "counts.mtx")), "CsparseMatrix")
rownames(m) <- readLines(file.path(td, "genes.txt")); colnames(m) <- readLines(file.path(td, "cells.txt"))
obj <- CreateInfercnvObject(raw_counts_matrix = m, gene_order_file = gof, annotations_file = file.path(td, "annotations.txt"),
                            delim = "\t", ref_group_names = refs)
set.seed(seed)
t0 <- proc.time()[["elapsed"]]
res <- infercnv::run(obj, cutoff = cutoff, out_dir = file.path(od, "work"), cluster_by_groups = TRUE, denoise = TRUE,
                     HMM = TRUE, no_plot = TRUE, num_threads = 1, BayesMaxPNormal = 0)
el <- proc.time()[["elapsed"]] - t0
w <- file.path(od, "work")
pick <- function(pat) { f <- list.files(w, pattern = pat, full.names = TRUE); f[!grepl("\\.(dat|cell_groupings)$", f)][1] }
dump <- function(o, tag) { x <- as.matrix(o@expr.data); writeBin(as.vector(x), file.path(od, paste0(tag, ".bin")))
  writeLines(rownames(x), file.path(od, paste0(tag, "_genes.txt"))); writeLines(colnames(x), file.path(od, paste0(tag, "_cells.txt"))) }
for (s in c("^14_invert", "^17_HMM_pred.*infercnv_obj$", "^20_HMM_pred", "^22_denoise")) { f <- pick(s); if (!is.na(f)) dump(readRDS(f), substr(s, 2, 3)) }
o <- readRDS(pick("^15_tumor_subclusters")); sc <- o@tumor_subclusters$subclusters
rows <- do.call(rbind, lapply(names(sc), function(g) do.call(rbind, lapply(names(sc[[g]]), function(s)
  data.frame(cell = colnames(o@expr.data)[sc[[g]][[s]]], group = g, subcluster = s)))))
write.table(rows, file.path(od, "subclusters.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
unlink(w, recursive = TRUE)
write_json(list(engine = "infercnv", version = as.character(packageVersion("infercnv")), seed = seed, cutoff = cutoff,
                dim = dim(res@expr.data), seconds = el), file.path(od, "result.json"), auto_unbox = TRUE, digits = NA)
cat(basename(td), sprintf("%d genes x %d cells, %.1f s\n", nrow(res@expr.data), ncol(res@expr.data), el))

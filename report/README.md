# Benchmark: pyinfercnv vs R inferCNV 1.26.0

Seven datasets: the oligodendroglioma example bundled with inferCNV (Smart-seq2, 184 cells; reference:
microglia/macrophages and non-malignant oligodendrocytes) and six breast tumours from 10x Genomics
(GSE148673: DCIS1, TNBC1–5; 532–3,225 cells; reference: the cells CopyKAT called normal in the GEO files).

```
python  scripts/01_prepare.py <R infercnv extdata dir>       # GEO files + the oligodendroglioma example -> tasks/
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 12 1,2   # idle cores, 1 thread each
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures
```

Both run `infercnv::run(cutoff, cluster_by_groups = TRUE, denoise = TRUE, HMM = TRUE, BayesMaxPNormal = 0,
num_threads = 1)` — cutoff 1 for Smart-seq2 and 0.1 for 10x — after `set.seed(1)`; pyinfercnv runs the same
configuration with `random_state=1`. R: inferCNV 1.26.0 with Seurat 5.5.1 and igraph 2.3.4 on R 4.5.3
(bioconda). Times are the faster of two runs, each on an idle core with one BLAS thread.
`figures/comparison_per_dataset.tsv` holds the per-dataset numbers.

# Stored R values for tests/unit/test_rcompat.py. Run from the repository root with
#   OPENBLAS_CORETYPE=Nehalem Rscript tests/data/rcompat/make_rcompat_ref.R
# (BLAS without FMA kernels, so smooth.spline's banded solve is the reference arithmetic).
suppressPackageStartupMessages({library(igraph); library(Matrix)})
o <- "tests/data/rcompat"
set.seed(11); writeBin(c(rnorm(2000), rnorm(4, c(1, 2, 3, 4), c(0, 1, 0, 2)), runif(10), sample(1e5, 50, replace = TRUE)), file.path(o, "rng.bin"))
set.seed(5); n <- 700; x <- c(rep(0, 120), round(rexp(n - 120), 4)); y <- log1p(x) * 1.3 + rnorm(n, sd = 0.2 + 0.1 * x)
f <- smooth.spline(x, y); g <- seq(-0.5, max(x) + 0.5, length.out = 301)
writeBin(c(x, y), file.path(o, "spline_xy.bin")); writeBin(c(f$spar, f$lambda, predict(f, g)$y), file.path(o, "spline_fit.bin"))
set.seed(3); n <- 120; x <- matrix(rnorm(n * 5), n); x[1:40, 1] <- x[1:40, 1] + 3; x[41:80, 2] <- x[41:80, 2] + 3
d <- as.matrix(dist(x)); nn <- t(apply(d, 1, function(r) order(r)[1:15]))
A <- sparseMatrix(i = rep(1:n, each = 15), j = as.vector(t(nn)), x = 1, dims = c(n, n)); S <- A %*% t(A); S@x <- S@x / (30 - S@x); S@x[S@x < 1/15] <- 0; S <- drop0(S)
gr <- graph_from_adjacency_matrix(S, mode = "min", weighted = TRUE)
write.table(cbind(as_edgelist(gr, names = FALSE), E(gr)$weight), file.path(o, "leiden_edges.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
set.seed(7); p <- cluster_leiden(gr, objective_function = "CPM", resolution = 0.08); writeLines(as.character(p$membership), file.path(o, "leiden_membership.txt"))

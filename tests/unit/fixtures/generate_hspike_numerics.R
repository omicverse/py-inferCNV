# Synthetic numerical oracle; no patient data. R 4.5.3, stats 4.5.3.
# Run from this directory: Rscript --vanilla generate_hspike_numerics.R
options(digits = 17)
set.seed(20261022, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
u <- runif(64)
z <- rnorm(64)
idx <- sample.int(4056, 64, replace = TRUE) - 1L
singleton <- sample.int(1, 64, replace = TRUE) - 1L
tail <- runif(64)
write.table(data.frame(u, z, idx, singleton, tail), "hspike_rng.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE)
x <- rep(seq(0.2, 6, length.out = 90), 1L + seq_len(90) %% 4L)
y <- sin(1.3 * x) + 0.12 * cos(17 * x) + 0.05 * sin(seq_along(x))
fit <- smooth.spline(x, y)
at <- seq(-1, 8, length.out = 121)
write.table(data.frame(x, y), "hspike_spline_input.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(x = at, y = predict(fit, at)$y), "hspike_spline_prediction.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(spar = fit$spar, df = fit$df), "hspike_spline_fit.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE)

# Changelog

## 0.1.0.dev0 (unreleased)

### Added
- Phase 1: preprocess (filter, CPM normalize, log2, subtract reference with bounded logic, max-centered threshold)
- Phase 1: smoothing (scipy.ndimage.uniform_filter1d × 2 + numba tail helper) per chromosome
- Phase 1: per-cell median centering
- Phase 1: outlier pruning
- Phase 1: invert log2
- Phase 1: AnnData-native API (`counts_layer="counts"` default)
- Phase 1: typer CLI (`pyinfercnv run-h5ad`)
- Phase 1: matplotlib heatmap visualization (no omicverse dependency)
- Phase 1: gene reference parquets shipped for hg38 / hg19 / mm10
- Phase 1: R-parity test suite (mixed tier-4 bit-exact / tier-4 approximate)
- Phase 1: benchmark scripts reused from pycopykat template

### Parity
- See README.md "Parity status" section.

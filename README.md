# pyinfercnv

Pure-Python re-implementation of [inferCNV](https://github.com/broadinstitute/inferCNV) (Broad Institute) — single-cell CNV inference from scRNA-seq, AnnData-native, R-parity-audited.

**Status:** v0.1.0.dev0 — Phase 1 (preprocess → smooth → outlier prune). Phase 2 (HMM + subclustering) and Phase 3 (BayesNet + denoise) in development.

## Installation

```bash
pip install pyinfercnv              # core
pip install 'pyinfercnv[viz]'       # + matplotlib for heatmap
pip install 'pyinfercnv[compare]'   # + hmmlearn for parity cross-check
```

## Quickstart

See `examples/tutorial_phase1.ipynb`.

## Parity status

TBD (filled at end of Phase 1 by Task 43).

## License

BSD-3-Clause, matching upstream R inferCNV.

"""6-state (i6) and 3-state (i3) HMM modules for pyinfercnv.

Mirrors R `inferCNV_HMM.R` (i6) and `inferCNV_i3HMM.R` (i3). Pure-Python +
numba — NO hmmlearn in runtime dependencies (hmmlearn is a `[compare]`
extra only and only imported inside tests via `pytest.importorskip`).
"""
from __future__ import annotations

from pyinfercnv.hmm.i3 import estimate_i3_state_params, predict_i3
from pyinfercnv.hmm.i6 import predict_i6

__all__ = [
    "predict_i6",
    "predict_i3",
    "estimate_i3_state_params",
]

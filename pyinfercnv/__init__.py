"""pyinfercnv — pure-Python re-implementation of Broad Institute R infercnv."""
from __future__ import annotations

from pyinfercnv.config import InferCNVConfig
from pyinfercnv.pipeline import infercnv
from pyinfercnv.pipeline_phase3 import run_phase3  # WIP skeleton; raises when toggles on
from pyinfercnv.result import InferCNVResult

__version__ = "0.2.0.dev2"

__all__ = [
    "InferCNVConfig",
    "InferCNVResult",
    "infercnv",
    "run_phase3",  # WIP: see pipeline_phase3 docstring
    "__version__",
]

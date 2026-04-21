"""Numba hot kernels — only for loop-bound code that scipy cannot express."""
from pyinfercnv.kernels.smooth_tail_numba import smooth_tail_inplace, smooth_tail_overwrite

__all__ = ["smooth_tail_inplace", "smooth_tail_overwrite"]

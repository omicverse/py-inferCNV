"""Hspike must honor the same pruning switch as the real expression matrix."""
import numpy as np

from pyinfercnv.config import InferCNVConfig
from pyinfercnv.hmm.hspike import _run_phase1_replay_on_hspike


def test_hspike_replay_respects_prune_setting():
    rng = np.random.default_rng(91)
    counts = rng.poisson(8, size=(20, 200)).astype(np.float32)
    counts[10:, :100] *= 4
    counts[10:, 100:] *= 0.1
    chroms = np.array(["chrA"] * 100 + ["chr_B"] * 100, dtype=object)
    refs = {"normal": np.arange(10)}
    common = dict(window_length=31, outlier_lower_bound=0.98, outlier_upper_bound=1.02)
    unpruned, _ = _run_phase1_replay_on_hspike(
        counts, gene_chr_labels=chroms, reference_indices=refs,
        config=InferCNVConfig(prune_outliers=False, **common),
    )
    pruned, _ = _run_phase1_replay_on_hspike(
        counts, gene_chr_labels=chroms, reference_indices=refs,
        config=InferCNVConfig(prune_outliers=True, **common),
    )
    assert unpruned.min() < 0.98 and unpruned.max() > 1.02
    np.testing.assert_allclose(pruned, np.clip(unpruned, 0.98, 1.02), rtol=0, atol=1e-7)

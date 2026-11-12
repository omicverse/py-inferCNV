"""Parallel permutations must preserve the serial random stream and tree."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np
import pytest

from pyinfercnv.subcluster.random_trees import _parameterize_null, random_tree_subcluster


@pytest.mark.parametrize("cached_uint32", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parallel_matches_serial_random_stream(cached_uint32, dtype):
    X = np.random.default_rng(12).normal(size=(36, 51)).astype(dtype)
    serial_rng = np.random.default_rng(42)
    parallel_rng = np.random.default_rng(42)
    if cached_uint32:
        serial_rng.integers(2**32, dtype=np.uint32)
        parallel_rng.integers(2**32, dtype=np.uint32)
    kwargs = dict(hclust_method="ward", window_size=11, n_rand_iters=6)
    serial = _parameterize_null(X, rng=serial_rng, **kwargs)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        parallel = _parameterize_null(X, rng=parallel_rng, executor=pool, **kwargs)
    assert np.array_equal(serial[0], parallel[0])
    assert np.array_equal(serial[1], parallel[1])
    assert serial[2] == parallel[2]
    assert serial_rng.bit_generator.state == parallel_rng.bit_generator.state
    assert np.array_equal(serial_rng.integers(2**32, size=8, dtype=np.uint32),
                          parallel_rng.integers(2**32, size=8, dtype=np.uint32))


@pytest.mark.parametrize("workers", [2, 3])
@pytest.mark.parametrize("subsample", [500, 17])
def test_parallel_recursion_matches_serial(workers, subsample):
    X = np.random.default_rng(5).normal(size=(40, 51)).astype(np.float32)
    kwargs = dict(random_state=20261022, n_rand_iters=6, window_size=11,
                  subsample_for_tree=subsample)
    if subsample < len(X):
        with pytest.warns(UserWarning, match="subsample"):
            serial = random_tree_subcluster(X, **kwargs)
        with pytest.warns(UserWarning, match="subsample"):
            parallel = random_tree_subcluster(X, tree_workers=workers, **kwargs)
    else:
        serial = random_tree_subcluster(X, **kwargs)
        parallel = random_tree_subcluster(X, tree_workers=workers, **kwargs)
    assert np.array_equal(serial, parallel)


def test_config_workers_reach_the_partition_backend(monkeypatch):
    from pyinfercnv import InferCNVConfig
    from pyinfercnv.pipeline_phase2 import _run_subclustering

    seen = []

    def capture(matrix, *, tree_workers, random_state):
        seen.append((len(matrix), tree_workers, random_state))
        return np.zeros(len(matrix), dtype=np.int32)

    monkeypatch.setattr("pyinfercnv.subcluster.random_trees.random_tree_subcluster", capture)
    cfg = InferCNVConfig(tumor_subcluster_partition_method="random_trees",
                         tree_workers=3, cluster_by_groups=False)
    is_reference = np.array([True] * 5 + [False] * 15)
    _run_subclustering(np.ones((20, 11)), is_reference,
                       config=cfg, random_state=42, profile=None)
    assert seen == [(15, 3, 42)]


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_is_rejected(workers):
    from pyinfercnv import InferCNVConfig

    with pytest.raises(ValueError, match="tree_workers"):
        InferCNVConfig(tree_workers=workers).validate()
    with pytest.raises(ValueError, match="tree_workers"):
        random_tree_subcluster(np.ones((6, 11)), tree_workers=workers)


def test_worker_failure_releases_shared_memory():
    names = []

    class BrokenExecutor:
        def map(self, function, tasks):
            names.append(tasks[0][0])
            raise RuntimeError("synthetic worker failure")

    matrix = np.random.default_rng(6).normal(size=(12, 11))
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        _parameterize_null(matrix, hclust_method="ward", window_size=5,
                            n_rand_iters=2, rng=np.random.default_rng(42),
                            executor=BrokenExecutor())
    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[0])

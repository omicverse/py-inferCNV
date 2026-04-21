from tests.conftest import r_reference_available
from tests._r_parity_helpers import assert_bit_exact
import numpy as np


def test_assert_bit_exact_passes_identical():
    assert_bit_exact(np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32))


def test_assert_bit_exact_raises_on_diff():
    import pytest
    with pytest.raises(AssertionError, match="max_diff"):
        assert_bit_exact(np.array([0.0]), np.array([1e-3]))


def test_r_reference_available_returns_bool():
    result = r_reference_available("nonexistent")
    assert result is False

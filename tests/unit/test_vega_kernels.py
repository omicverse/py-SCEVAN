"""Bit-exact guard for the njit pre-kernel float32 reductions.

``_calc_std_kernel`` / ``_calc_mean_kernel`` (segment/_vega_core.py) replaced
pure-Python scalar loops for speed. Their output feeds vega_mc_kernel and
determines breakpoints, so they MUST be bit-identical to the original float32
sequential accumulation -- not merely close. The frozen pure-Python references
below are the literal pre-optimization bodies; the njit kernels are asserted
equal to them via np.array_equal (zero ULP), never allclose.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyscevan.segment._vega_core import _calc_mean_kernel, _calc_std_kernel


def _ref_calc_std(data_chr: np.ndarray, np_chr: int) -> np.ndarray:
    """Frozen literal of the original pure-Python _calc_std_per_sample."""
    num_samples = data_chr.shape[0]
    std = np.empty(num_samples, dtype=np.float32)
    for j in range(num_samples):
        v = data_chr[j]
        s = np.float32(0.0)
        for x in v:
            s = np.float32(s + x)
        m = np.float32(s / np_chr)
        acc = np.float32(0.0)
        for x in v:
            d = np.float32(x - m)
            acc = np.float32(acc + np.float32(d * d))
        std[j] = np.float32(np.sqrt(acc / np.float32(np_chr - 1)))
    return std


def _ref_calc_mean(v: np.ndarray, n: int) -> np.float32:
    """Frozen literal of the original pure-Python _calc_mean."""
    s = np.float32(0.0)
    for x in v:
        s = np.float32(s + x)
    return np.float32(s / n)


@pytest.mark.parametrize("seed", range(12))
def test_calc_std_kernel_bit_exact(seed):
    rng = np.random.default_rng(seed)
    num_samples = int(rng.integers(1, 9))
    np_chr = int(rng.integers(2, 61))  # >=2 probes (1-probe is guarded upstream)
    scale = 10.0 ** rng.integers(-4, 7)  # exercise float32 cancellation
    data = (rng.standard_normal((num_samples, np_chr)) * scale).astype(np.float32)
    got = _calc_std_kernel(np.ascontiguousarray(data), np_chr)
    ref = _ref_calc_std(data, np_chr)
    assert got.dtype == np.float32
    assert np.array_equal(got, ref), f"seed={seed} std bit mismatch"


def test_calc_std_kernel_edges():
    # np_chr == 2 (n-1 == 1 divisor)
    d2 = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    assert np.array_equal(_calc_std_kernel(d2, 2), _ref_calc_std(d2, 2))
    # flat / zero-variance chromosome -> std == 0 (cf. fixture c9)
    flat = np.zeros((3, 20), dtype=np.float32)
    got = _calc_std_kernel(flat, 20)
    assert np.array_equal(got, _ref_calc_std(flat, 20))
    assert np.all(got == np.float32(0.0))


@pytest.mark.parametrize("seed", range(12))
def test_calc_mean_kernel_bit_exact(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(1, 400))  # incl size==1 segments
    scale = 10.0 ** rng.integers(-4, 7)
    v = (rng.standard_normal(n) * scale).astype(np.float32)
    got = _calc_mean_kernel(v, n)
    ref = _ref_calc_mean(v, n)
    # numba boxes the float32 scalar return as a float64 (value-preserving); the
    # call site re-wraps in np.float32(), so the contract is float32-cast bit
    # equality, which is what downstream `np.float32(acc + m)` consumes.
    assert np.float32(got).tobytes() == np.float32(ref).tobytes(), (
        f"seed={seed} mean bit mismatch"
    )

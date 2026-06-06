"""White-box unit tests for the vegaMC Numba kernel internals.

Guards the three trickiest pieces before the R-parity suite:
  (a) init_trivial_segmentation -- ROUND_CNST floor on probe 0 only + lambda cost
  (b) update_priority -- Ward cost on a hand example
  (c) heap compare_priority tie-break -- equal priority => smaller id wins
"""

import numpy as np

from pyscevan.segment._vega_core import (
    _compare_priority,
    _heap_extract_max,
    _heap_insert,
    _init_trivial,
    _update_priority,
)


def test_init_trivial_round_cnst_and_lambda():
    # 3 probes x 2 samples.
    #   sample0: 0.8, 0.6, 0.6
    #   sample1: 0.8, 0.6, 0.6
    # data is indexed [sample, probe].
    data = np.array([[0.8, 0.6, 0.6], [0.8, 0.6, 0.6]], dtype=np.float32)
    num_probes = 3
    num_samples = 2
    weight = np.ones(num_samples, dtype=np.float32)

    brk_p = np.zeros(num_probes + 1, dtype=np.float32)
    brk_size = np.zeros(num_probes + 1, dtype=np.int64)
    brk_prev = np.zeros(num_probes + 1, dtype=np.int64)
    brk_next = np.zeros(num_probes + 1, dtype=np.int64)
    brk_sum = np.zeros((num_probes + 1, num_samples), dtype=np.float32)
    heap_p = np.zeros(num_probes + 1, dtype=np.float32)
    heap_id = np.zeros(num_probes + 1, dtype=np.int64)

    heap_size = _init_trivial(
        data, num_probes, num_samples, weight,
        brk_p, brk_size, brk_prev, brk_next, brk_sum,
        heap_p, heap_id,
    )

    # probe 0: ROUND_CNST floor(0.8*1000)/1000 == 0.8 for both samples
    assert brk_sum[0, 0] == np.float32(np.floor(np.float32(0.8) * 1000) / 1000)
    assert brk_sum[0, 1] == brk_sum[0, 0]
    # probe 0 sum != raw 0.8 only if floor matters; here it floors to 0.800
    np.testing.assert_allclose(brk_sum[0, 0], 0.8, atol=1e-6)

    # probes 1,2 sums are RAW (not rounded): 0.6
    np.testing.assert_allclose(brk_sum[1, 0], 0.6, atol=1e-6)
    np.testing.assert_allclose(brk_sum[2, 0], 0.6, atol=1e-6)

    # closure sentinel at index num_probes
    assert brk_size[num_probes] == -1
    assert brk_prev[num_probes] == -1
    assert brk_next[num_probes] == -1

    # lambda[1] = -0.5 * ((0.8-0.6)^2 + (0.8-0.6)^2) = -0.5 * (0.04+0.04) = -0.04
    np.testing.assert_allclose(brk_p[1], -0.04, atol=1e-6)
    # lambda[2] = -0.5 * (0 + 0) = 0
    np.testing.assert_allclose(brk_p[2], 0.0, atol=1e-7)

    # links: 0 ->1 ->2 -> (3=closure)
    assert brk_next[0] == 1 and brk_prev[0] == -1
    assert brk_prev[1] == 0 and brk_next[1] == 2
    assert brk_prev[2] == 1 and brk_next[2] == 3
    assert brk_size[0] == 1 and brk_size[1] == 1 and brk_size[2] == 1

    # heap holds probes 1 and 2 (capacity num_probes-1 == 2)
    assert heap_size == 2
    # max priority node is probe 2 (p=0 > p=-0.04)
    max_p, max_id, _ = _heap_extract_max(heap_p, heap_id, heap_size)
    assert max_id == 2
    np.testing.assert_allclose(max_p, 0.0, atol=1e-7)


def test_update_priority_ward_formula():
    # 2 breakpoints: prev=id0 (size 2, sum [1,2]); curr=id1 (size 1, sum [0,1]).
    num_samples = 2
    weight = np.ones(num_samples, dtype=np.float32)
    brk_sum = np.zeros((3, num_samples), dtype=np.float32)
    brk_size = np.zeros(3, dtype=np.int64)
    brk_prev = np.zeros(3, dtype=np.int64)

    brk_sum[0] = [1.0, 2.0]
    brk_size[0] = 2
    brk_sum[1] = [0.0, 1.0]
    brk_size[1] = 1
    brk_prev[1] = 0

    # prev_mean = [0.5, 1.0]; curr_mean = [0.0, 1.0]
    # second = (0.5)^2 + (0.0)^2 = 0.25
    # first  = (2*1)/(2+1) = 2/3
    # return -(2/3 * 0.25) = -1/6
    tl = _update_priority(brk_sum, brk_size, brk_prev, weight, num_samples, 1)
    np.testing.assert_allclose(tl, -(2.0 / 3.0) * 0.25, atol=1e-6)

    # i == 0 -> -1.0 sentinel
    assert _update_priority(brk_sum, brk_size, brk_prev, weight, num_samples, 0) == np.float32(-1.0)


def test_compare_priority_tiebreak_smaller_id_wins():
    p = np.float32(-0.5)
    # equal priority: smaller id wins
    assert _compare_priority(p, 3, p, 7) is True
    assert _compare_priority(p, 7, p, 3) is False
    # equal id and priority -> not strictly greater
    assert _compare_priority(p, 5, p, 5) is False
    # higher priority wins regardless of id
    assert _compare_priority(np.float32(0.0), 99, np.float32(-1.0), 1) is True
    assert _compare_priority(np.float32(-1.0), 1, np.float32(0.0), 99) is False


def test_heap_tiebreak_extract_order():
    # insert three nodes with EQUAL priority, ids 5, 2, 8.
    # extract order must be ascending id (2,5,8) since smaller id wins ties.
    heap_p = np.zeros(8, dtype=np.float32)
    heap_id = np.zeros(8, dtype=np.int64)
    hs = 0
    p = np.float32(-0.3)
    for nid in (5, 2, 8):
        hs = _heap_insert(heap_p, heap_id, hs, p, nid)
    got = []
    while hs > 0:
        mp, mid, hs = _heap_extract_max(heap_p, heap_id, hs)
        got.append(mid)
    assert got == [2, 5, 8]

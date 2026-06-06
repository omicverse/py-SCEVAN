"""Numba @njit kernel mirroring SCEVAN-R/src/vegaMC.c.

Array-based linked-list of breakpoints + a 1-indexed binary heap, in float32
storage (float64 only as transient pow/sqrt intermediates, then stored back to
float32 -- the C casts ``pow()``/``sqrt()`` double results into ``float`` vars,
mimicked here).

Layout vs vegaMC.c:
    * brks linked list -> per-field NumPy arrays (brk_p, brk_size, brk_prev,
      brk_next) + a 2-D brk_sum[id, sample].  ``brks[num_probes]`` is the
      closure sentinel (size=-1, prev=next=num_probes-equivalent, id=num_probes).
    * binary heap -> heap_p (float32) + heap_id (int) arrays, 1-indexed
      (slot 0 unused), plus a scalar heap_size.

Everything is ported function-by-function; non-obvious bits cite the C line.
"""

import numpy as np
from numba import njit

EPSILON = np.float32(0.01)
ROUND_CNST = np.float32(1000.0)


# ----------------------------------------------------------------------------
# Heap: compare_priority (vegaMC.c:346-358)
#   i.p > j.p -> 1 ; i.p < j.p -> 0 ; equal -> (i.id < j.id) ? 1 : 0
#   i.e. higher priority wins, ties broken by SMALLER id.
# ----------------------------------------------------------------------------
@njit(cache=True)
def _compare_priority(pi, idi, pj, idj):
    if pi > pj:
        return True
    elif pi < pj:
        return False
    else:
        return idi < idj


# heapify (vegaMC.c:126-155): check LEFT child before RIGHT.
@njit(cache=True)
def _heapify(heap_p, heap_id, heap_size, i):
    while True:
        lc = 2 * i
        rc = 2 * i + 1
        largest = i
        # check left child (vegaMC.c:136-137)
        if lc <= heap_size and _compare_priority(
            heap_p[lc], heap_id[lc], heap_p[largest], heap_id[largest]
        ):
            largest = lc
        # check right child (vegaMC.c:142-144)
        if rc <= heap_size and _compare_priority(
            heap_p[rc], heap_id[rc], heap_p[largest], heap_id[largest]
        ):
            largest = rc
        if largest != i:
            tp = heap_p[i]
            ti = heap_id[i]
            heap_p[i] = heap_p[largest]
            heap_id[i] = heap_id[largest]
            heap_p[largest] = tp
            heap_id[largest] = ti
            i = largest  # iterative recursion (vegaMC.c:153)
        else:
            return


# heap_insert (vegaMC.c:195-227).  Returns new heap_size.
@njit(cache=True)
def _heap_insert(heap_p, heap_id, heap_size, key_p, key_id):
    heap_size += 1
    i = heap_size
    # bubble up while compare_priority(key, parent) (vegaMC.c:218)
    while i > 1 and _compare_priority(key_p, key_id, heap_p[i // 2], heap_id[i // 2]):
        heap_p[i] = heap_p[i // 2]
        heap_id[i] = heap_id[i // 2]
        i = i // 2
    heap_p[i] = key_p
    heap_id[i] = key_id
    return heap_size


# heap_extract_max (vegaMC.c:169-187).  Returns (max_p, max_id, new_heap_size).
@njit(cache=True)
def _heap_extract_max(heap_p, heap_id, heap_size):
    max_p = heap_p[1]
    max_id = heap_id[1]
    heap_p[1] = heap_p[heap_size]
    heap_id[1] = heap_id[heap_size]
    heap_size -= 1
    if heap_size >= 1:
        _heapify(heap_p, heap_id, heap_size, 1)
    return max_p, max_id, heap_size


# heap_delete (vegaMC.c:233-255).  i is a 1-indexed slot.  Returns new heap_size.
@njit(cache=True)
def _heap_delete(heap_p, heap_id, heap_size, i):
    heap_p[i] = heap_p[heap_size]
    heap_id[i] = heap_id[heap_size]
    heap_size -= 1
    if i <= heap_size:
        _heapify(heap_p, heap_id, heap_size, i)
    return heap_size


# node_find (vegaMC.c:363-371): linear scan by id, return slot or -1 (FAILED).
@njit(cache=True)
def _node_find(heap_id, heap_size, target_id):
    for i in range(1, heap_size + 1):
        if heap_id[i] == target_id:
            return i
    return -1


# ----------------------------------------------------------------------------
# update_priority (vegaMC.c:553-587): Ward-style cost for breakpoint i.
#   i == 0 -> -1.0
#   else: prev = brk_prev[i];
#     second = sum_j (|prev_mean_j - curr_mean_j|)^2 * weight[j]
#     first  = (prev_size*curr_size)/(prev_size+curr_size)
#     return -(first*second)
# Stored float32; pow uses float64 transient then cast (C casts pow() to float).
# ----------------------------------------------------------------------------
@njit(cache=True)
def _update_priority(brk_sum, brk_size, brk_prev, weight, num_samples, i):
    if i == 0:
        return np.float32(-1.0)
    prev_id = brk_prev[i]
    prev_size = brk_size[prev_id]
    curr_size = brk_size[i]
    second = np.float32(0.0)
    for j in range(num_samples):
        prev_mean = np.float32(brk_sum[prev_id, j] / prev_size)
        curr_mean = np.float32(brk_sum[i, j] / curr_size)
        tmp = np.float32(prev_mean - curr_mean)
        if tmp < 0:
            tmp = np.float32(-tmp)
        # (float) pow(tmp, 2) * weight[j]  (vegaMC.c:577)
        second = np.float32(second + np.float32(tmp * tmp) * weight[j])
    first = np.float32(
        np.float32(prev_size * curr_size) / np.float32(prev_size + curr_size)
    )
    lam = np.float32(first * second)
    return np.float32(-lam)


# ----------------------------------------------------------------------------
# init_trivial_segmentation (vegaMC.c:590-647)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _init_trivial(
    data, num_probes, num_samples, weight,
    brk_p, brk_size, brk_prev, brk_next, brk_sum,
    heap_p, heap_id,
):
    # brks[0]: ROUND_CNST floor applied ONLY to probe 0 (vegaMC.c:602-607)
    for j in range(num_samples):
        brk_sum[0, j] = np.float32(
            np.floor(data[j, 0] * ROUND_CNST) / ROUND_CNST
        )
    brk_p[0] = np.float32(-1.0)
    brk_size[0] = 1
    brk_prev[0] = -1  # C: unsigned wrap, unused
    brk_next[0] = 1

    # closure sentinel brks[num_probes] (vegaMC.c:609-616)
    for j in range(num_samples):
        brk_sum[num_probes, j] = np.float32(-1.0)
    brk_p[num_probes] = np.float32(-1.0)
    brk_size[num_probes] = -1
    brk_prev[num_probes] = -1
    brk_next[num_probes] = -1

    heap_size = 0
    # all other nodes i = 1 .. num_probes-1 (vegaMC.c:621-646)
    for i in range(1, num_probes):
        tmp_lambda = np.float32(0.0)
        for j in range(num_samples):
            data_prev = data[j, i - 1]
            data_curr = data[j, i]
            brk_sum[i, j] = np.float32(data_curr)  # raw, NOT rounded
            a = np.float32(data_prev - data_curr)
            if a < 0:
                a = np.float32(-a)
            c = np.float32(a * a)  # pow(a,2)
            tmp_lambda = np.float32(tmp_lambda + c * weight[j])
        tmp_lambda = np.float32(np.float32(0.5) * tmp_lambda)
        brk_p[i] = np.float32(-tmp_lambda)
        brk_size[i] = 1
        brk_prev[i] = i - 1
        brk_next[i] = i + 1
        heap_size = _heap_insert(heap_p, heap_id, heap_size, np.float32(-tmp_lambda), i)
    return heap_size


# ----------------------------------------------------------------------------
# vegaMC core (vegaMC.c:422-546).  data shape (num_samples, num_probes) float32.
# markers_start: int array, global probe idx of each local probe.
# Returns (out_start_idx, out_end_idx, out_size, out_mean, n_reg) -- arrays sized
# to n_reg.  out_start_idx/out_end_idx are GLOBAL probe indices (markers_start).
# ----------------------------------------------------------------------------
@njit(cache=True)
def vega_mc_kernel(data, markers_start, beta_value, std, num_samples, weight, weight_sum):
    num_probes = data.shape[1]

    # stop_lambda_gradient (vegaMC.c:441-446)
    tmp_sd = np.float32(0.0)
    for j in range(num_samples):
        tmp_sd = np.float32(tmp_sd + std[j] * weight[j])
    stop = np.float32(np.float32(tmp_sd / weight_sum) * beta_value)

    # breakpoint arrays: index 0..num_probes (closure at num_probes)
    brk_p = np.zeros(num_probes + 1, dtype=np.float32)
    brk_size = np.zeros(num_probes + 1, dtype=np.int64)
    brk_prev = np.zeros(num_probes + 1, dtype=np.int64)
    brk_next = np.zeros(num_probes + 1, dtype=np.int64)
    brk_sum = np.zeros((num_probes + 1, num_samples), dtype=np.float32)

    # heap capacity = num_probes - 1 (vegaMC.c:451); 1-indexed -> +1 slot.
    heap_p = np.zeros(num_probes + 1, dtype=np.float32)
    heap_id = np.zeros(num_probes + 1, dtype=np.int64)

    heap_size = _init_trivial(
        data, num_probes, num_samples, weight,
        brk_p, brk_size, brk_prev, brk_next, brk_sum,
        heap_p, heap_id,
    )

    # lambda = heap_max().p - EPSILON ; lambda_gradient = |lambda|
    lam = np.float32(heap_p[1] - EPSILON)
    lambda_gradient = np.float32(abs(lam))

    # outer loop (vegaMC.c:461)
    while heap_size > 0 and lambda_gradient <= stop:
        # inner loop (vegaMC.c:464)
        while heap_size > 0 and heap_p[1] > lam:
            max_p, max_id, heap_size = _heap_extract_max(heap_p, heap_id, heap_size)

            # stale-node reinsert (vegaMC.c:468-470)
            if (max_p > brk_p[max_id]) and (brk_p[max_id] < lam):
                heap_size = _heap_insert(
                    heap_p, heap_id, heap_size, brk_p[max_id], max_id
                )
            else:
                # MERGE brks[max_id] into brks[prev] (vegaMC.c:472-512)
                d_prev = brk_prev[max_id]
                d_next = brk_next[max_id]
                d_size = brk_size[max_id]

                # unlink
                brk_next[d_prev] = d_next
                brk_prev[d_next] = d_prev
                # accumulate sum
                for j in range(num_samples):
                    brk_sum[d_prev, j] = np.float32(brk_sum[d_prev, j] + brk_sum[max_id, j])
                brk_size[d_prev] = brk_size[d_prev] + d_size

                tmp_lambda = _update_priority(
                    brk_sum, brk_size, brk_prev, weight, num_samples, d_prev
                )
                # if d_prev > 0: find/delete/reinsert node (vegaMC.c:490-496)
                if d_prev > 0:
                    node_index = _node_find(heap_id, heap_size, d_prev)
                    saved_id = heap_id[node_index]
                    heap_size = _heap_delete(heap_p, heap_id, heap_size, node_index)
                    heap_size = _heap_insert(
                        heap_p, heap_id, heap_size, tmp_lambda, saved_id
                    )
                brk_p[d_prev] = tmp_lambda  # set unconditionally (vegaMC.c:497)

                # d_next side (vegaMC.c:502-511)
                if d_next < num_probes:
                    tmp_lambda = _update_priority(
                        brk_sum, brk_size, brk_prev, weight, num_samples, d_next
                    )
                    node_index = _node_find(heap_id, heap_size, d_next)
                    saved_id = heap_id[node_index]
                    heap_size = _heap_delete(heap_p, heap_id, heap_size, node_index)
                    heap_size = _heap_insert(
                        heap_p, heap_id, heap_size, tmp_lambda, saved_id
                    )
                # brks[d_next].p = tmp_lambda set unconditionally, even for closure
                # (vegaMC.c:511)
                brk_p[d_next] = tmp_lambda

        # update lambda (vegaMC.c:519-522)
        if heap_size > 0:
            lambda_gradient = np.float32(lam - heap_p[1])
            lam = np.float32(heap_p[1] - EPSILON)

    # segment emission (vegaMC.c:526-543)
    # first count regions by walking the list
    n_reg = 0
    i = 0
    while i < num_probes:
        n_reg += 1
        i = brk_next[i]

    out_start = np.empty(n_reg, dtype=np.int64)
    out_end = np.empty(n_reg, dtype=np.int64)
    out_size = np.empty(n_reg, dtype=np.int64)
    out_mean = np.empty(n_reg, dtype=np.float32)

    i = 0
    jj = 0
    while i < num_probes:
        nxt = brk_next[i]
        # C uses brk_del.id as the start probe; brks[i] is stored at array index
        # == its own id (we never relabel), so id == i.  end = next-1 (inclusive)
        # (vegaMC.c:529-530).
        out_start[jj] = markers_start[i]
        out_end[jj] = markers_start[nxt - 1]
        out_size[jj] = brk_size[i]
        tmp_sd = np.float32(0.0)
        for k in range(num_samples):
            tmp_sd = np.float32(
                tmp_sd
                + np.float32(brk_sum[i, k] / np.float32(brk_size[i])) * weight[k]
            )
        out_mean[jj] = np.float32(tmp_sd / weight_sum)
        i = nxt
        jj += 1

    return out_start, out_end, out_size, out_mean, n_reg


# ----------------------------------------------------------------------------
# Pre-kernel float32 reductions (calc_std / calc_mean, run_vegaMC.c:666-684).
# These run BEFORE vega_mc_kernel: the per-chromosome std feeds stop_lambda and
# DETERMINES breakpoints, so the float32 SEQUENTIAL accumulation order is load-
# bearing. njit transcribes the exact op sequence of the pure-Python originals
# (segment/vegamc.py) -- typed float32 accumulator, np.float32() after each op,
# float32/int -> float64 -> cast for the divisions -- which is the same parity-
# verified idiom as _update_priority / _init_trivial above. Bit-identical to the
# pure-Python reference (frozen in tests/unit/test_vega_kernels.py).
# ----------------------------------------------------------------------------
@njit(cache=True)
def _calc_std_kernel(data_chr, np_chr):
    """Per-sample std over a chromosome (calc_std, run_vegaMC.c:676-684).

    ``data_chr`` shape (num_samples, np_chr) float32. Replicates the float32
    sequential mean+SSE accumulation op-for-op.
    """
    num_samples = data_chr.shape[0]
    std = np.empty(num_samples, dtype=np.float32)
    for j in range(num_samples):
        s = np.float32(0.0)
        for k in range(np_chr):
            s = np.float32(s + data_chr[j, k])
        m = np.float32(s / np_chr)
        acc = np.float32(0.0)
        for k in range(np_chr):
            d = np.float32(data_chr[j, k] - m)
            acc = np.float32(acc + np.float32(d * d))
        std[j] = np.float32(np.sqrt(acc / np.float32(np_chr - 1)))
    return std


@njit(cache=True)
def _calc_mean_kernel(v, n):
    """sum(v)/n in float32 sequential accumulation (calc_mean, run_vegaMC.c:666)."""
    s = np.float32(0.0)
    for k in range(v.shape[0]):
        s = np.float32(s + v[k])
    return np.float32(s / n)

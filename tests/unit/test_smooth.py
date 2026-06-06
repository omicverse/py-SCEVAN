"""Isolated bit-exactness test for nonlinear_smooth vs a literal transcription
of SCEVAN's R nonLinSmooth loop (classifyTumor.R:201-214). Backs the README
'nonlinear_smooth bit-exact' parity claim with a direct assertion."""

import numpy as np

from pyscevan.classify.smooth import nonlinear_smooth

_ALPHA = 0.5
_DELTA_T = 0.2
_NITERS = 100


def _r_smooth_literal(col):
    """Literal per-column transcription of R nonLinSmooth (classifyTumor.R:201-214)."""
    y = np.concatenate([[0.0], col.astype(np.float64)])  # append(0, y)
    for _ in range(_NITERS):
        delta_p = (y[1:] - y[:-1]) / _ALPHA            # (y[2:]-y[:-1])/alpha
        delta_p = np.concatenate([delta_p, [0.0]])     # c(DeltaP, 0)
        t_d = np.tanh(delta_p)
        delta_m = t_d[1:] - t_d[:-1]                   # tD[2:]-tD[:-1]
        delta_m = np.concatenate([[0.0], delta_m])     # c(0, DeltaM)
        y = y + _DELTA_T * delta_m
    return y[1:]                                       # y[2:length(y)]


def test_nonlinear_smooth_matches_literal_r_loop():
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 0.4, size=(40, 6)).astype(np.float64)
    mat[10:20, :] -= 0.8  # a block, like a CNA region
    out = np.asarray(nonlinear_smooth(mat), dtype=np.float64)
    expected = np.column_stack([_r_smooth_literal(mat[:, c]) for c in range(mat.shape[1])])
    np.testing.assert_array_equal(out, expected)

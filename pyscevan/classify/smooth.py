"""Nonlinear (tanh) edge-preserving smoothing.

Port of the ``nonLinSmooth`` inner function in ``classifyTumor.R`` (lines
201-214).  Per cell column, runs an anisotropic-diffusion style smoother: the
relative-expression profile is treated as a 1-D signal indexed by gene order,
and a tanh-saturated gradient is iterated ``niters`` times.  This is the
SCEVAN "smooth data" step (step 8 of ``classifyTumorCells``).

The R recurrence, faithfully (1-based R indices):

    y <- append(0, y)                       # prepend a zero sentinel
    for i in 1:niters:
        DeltaP = (y[2:L] - y[1:(L-1)]) / alpha   # forward differences
        DeltaP = c(DeltaP, 0)                     # pad to length L
        tD     = tanh(DeltaP)
        DeltaM = tD[2:L] - tD[1:(L-1)]
        DeltaM = c(0, DeltaM)                     # pad-front to length L
        y      = y + DeltaT * DeltaM
    y <- y[2:length(y)]                     # drop the sentinel

with ``niters=100``, ``alpha=0.5``, ``DeltaT=0.2``.  float64.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nonlinear_smooth"]

_NITERS = 100
_ALPHA = 0.5
_DELTA_T = 0.2


def nonlinear_smooth(mat: np.ndarray) -> np.ndarray:
    """Apply the tanh nonlinear smoother to each column of ``mat``.

    Parameters
    ----------
    mat
        (genes x cells) array of relative expression. Each column is smoothed
        independently along the gene axis (row order is the genomic order).

    Returns
    -------
    np.ndarray
        Smoothed (genes x cells) array, float64, same shape as ``mat``.

    Notes
    -----
    Vectorised over cells; numerically identical (float64) to looping the R
    ``nonLinSmooth`` over columns. The sentinel zero prepended by R
    (``y <- append(0, y)``) shifts indices by one but, because ``DeltaM`` is
    front-padded with 0, the sentinel row never changes and is dropped at the
    end -- so the result equals iterating the recurrence on the bare column.
    The vectorised form below reproduces the per-column recurrence exactly by
    keeping the sentinel row 0 and operating on all columns at once.
    """
    y0 = np.asarray(mat, dtype=np.float64)
    n_genes, n_cells = y0.shape

    # y has a leading sentinel row of zeros (R: append(0, y)).
    y = np.empty((n_genes + 1, n_cells), dtype=np.float64)
    y[0, :] = 0.0
    y[1:, :] = y0
    L = n_genes + 1

    # Preallocate scratch reused across all 100 iterations (was re-zeroed every
    # iter). The two structurally-zero rows -- delta_p[L-1] and delta_m[0] -- are
    # written 0 once here and never overwritten by the in-place diffs (forward
    # diff fills [:L-1], backward diff fills [1:L]), exactly reproducing the
    # per-iter np.zeros. t_d is kept separate from delta_p (no aliasing). Pure
    # float64, same op order, same tanh -> bit-identical to the alloc-per-iter
    # form (guarded by tests/unit/test_smooth.py exact equality).
    delta_p = np.zeros((L, n_cells), dtype=np.float64)
    t_d = np.empty((L, n_cells), dtype=np.float64)
    delta_m = np.zeros((L, n_cells), dtype=np.float64)

    for _ in range(_NITERS):
        # DeltaP = (y[2:L] - y[1:(L-1)]) / alpha; last row stays 0.
        np.subtract(y[1:L, :], y[0 : L - 1, :], out=delta_p[: L - 1, :])
        delta_p[: L - 1, :] /= _ALPHA

        np.tanh(delta_p, out=t_d)

        # DeltaM = tD[2:L] - tD[1:(L-1)]; first row stays 0.
        np.subtract(t_d[1:L, :], t_d[0 : L - 1, :], out=delta_m[1:L, :])

        y += _DELTA_T * delta_m

    return y[1:, :]

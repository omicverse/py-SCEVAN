"""CNA matrix construction (port of SCEVAN::computeCNAmtx).

Mirrors ``classifyTumor.R`` lines 47-124. For each cell and each segment,
the per-segment mean of ``count_mtx`` is broadcast across the segment's rows
when that segment is flagged altered, else 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["compute_cna_mtx"]


def compute_cna_mtx(
    count_mtx: pd.DataFrame,
    breaks: np.ndarray,
    segm_alt: np.ndarray,
) -> pd.DataFrame:
    """Build the per-segment mean CNA matrix.

    Port of ``computeCNAmtx`` (classifyTumor.R:47-124).

    Parameters
    ----------
    count_mtx
        genes x cells DataFrame. Gene index and cell columns are preserved.
    breaks
        0-based breakpoint indices (sorted, incl 0 and ``n-1``), as returned
        by ``get_breaks_vegamc``. R uses 1-based breaks; the API here is
        0-based for consistency with the segmentation module.
    segm_alt
        Boolean array of length ``len(breaks) - 1`` (one per segment).
        ``True`` -> fill the segment with its per-cell mean; ``False`` -> 0.

    Returns
    -------
    pd.DataFrame
        CNA matrix (genes x cells), same index/columns as ``count_mtx``.

    Notes
    -----
    Breakpoint contract (R ``startBreaks=breaks[:-1]``,
    ``endBreaks=c(startBreaks[-1]-1, n)`` -- classifyTumor.R:51-53). In 0-based
    indexing, segment ``i`` spans rows ``startBreaks[i]..endBreaks[i]`` inclusive
    where ``startBreaks = breaks[:-1]`` and ``endBreaks[i] = breaks[i+1]-1`` for
    ``i < last``, with the FINAL segment ending at ``n-1`` inclusive.
    """
    breaks = np.asarray(breaks, dtype=int)
    segm_alt = np.asarray(segm_alt, dtype=bool)

    n = count_mtx.shape[0]
    start_breaks = breaks[:-1]
    end_breaks = breaks[1:] - 1
    end_breaks[-1] = n - 1  # final segment ends at n-1 (R: c(..., n))

    if len(segm_alt) != len(start_breaks):
        raise ValueError(
            f"segm_alt length {len(segm_alt)} != number of segments "
            f"{len(start_breaks)}"
        )

    values = count_mtx.to_numpy(dtype=float)
    cna = np.zeros_like(values, dtype=float)

    for i, (s, e) in enumerate(zip(start_breaks, end_breaks)):
        if segm_alt[i]:
            mean_value = values[s : e + 1, :].mean(axis=0)
        else:
            mean_value = 0.0
        cna[s : e + 1, :] = mean_value

    return pd.DataFrame(cna, index=count_mtx.index, columns=count_mtx.columns)

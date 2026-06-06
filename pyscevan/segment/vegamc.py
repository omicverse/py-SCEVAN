"""vegaMC orchestration layer (mirrors run_vegaMC.c call_VegaMC + driver).

Pure-Python driver around the Numba kernel in ``_vega_core``.  Splits the input
per chromosome (first-appearance order), computes per-chromosome per-sample std
(ddof=1), applies ``beta_eff = beta * num_samples`` once, runs the kernel per
chromosome, maps probe indices to genomic positions, and emits the 6
segmentation columns.

T4 (NOT implemented here): loss/gain % counts, "x%" strings, bootstrap
p-values, get_breaks_vegamc.
"""

import numpy as np
import pandas as pd

from ._vega_core import vega_mc_kernel


def vega_mc_r(
    mtx: pd.DataFrame,
    beta: float = 0.5,
    min_region_bp_size: int = 1000,
    loss_threshold: float = -0.2,  # noqa: ARG001 (T4)
    gain_threshold: float = 0.2,  # noqa: ARG001 (T4)
    with_pvalue: bool = False,
) -> pd.DataFrame:
    """Joint multi-sample segmentation (R-parity with SCEVAN vegaMC_R).

    Parameters
    ----------
    mtx
        DataFrame with columns ``[Name, Chr, Position, sample_1 .. sample_Ns]``
        (Name=col0, Chr=col1, Position=col2, samples=col3:).  Rows assumed
        grouped by chromosome (contiguous runs), as load_data assumes.
    beta
        Segmentation stop sensitivity (R ``beta``, default 0.5).
    min_region_bp_size
        Segments with bp ``Size <= min_region_bp_size`` are dropped (strict
        ``>`` kept, run_vegaMC.c:204).
    loss_threshold, gain_threshold
        Reserved for T4 (loss/gain % + p-values).  Unused in T3.
    with_pvalue
        Reserved for T4.  Raises if True.

    Returns
    -------
    DataFrame with columns ``["Chr","Start","End","Size","Mean","Probe Size"]``.
    """
    if with_pvalue:
        raise NotImplementedError("p-values land in T4")

    chrom = mtx["Chr"].to_numpy()
    pos = mtx["Position"].to_numpy().astype(np.int64)  # int cast (run_vegaMC.c:514)
    X = mtx.iloc[:, 3:].to_numpy().astype(np.float32)  # (n_probes, n_samples)
    n_probes, num_samples = X.shape

    # beta_eff = beta * num_samples, applied once (run_vegaMC.c:141)
    beta_eff = np.float32(np.float32(beta) * np.float32(num_samples))
    weight = np.ones(num_samples, dtype=np.float32)
    weight_sum = np.float32(num_samples)

    # contiguous chromosome runs in first-appearance order (run_vegaMC.c:509,591-601)
    chr_starts = []
    chr_ends = []
    prev = None
    for k in range(n_probes):
        if prev is None or chrom[k] != prev:
            chr_starts.append(k)
            if prev is not None:
                chr_ends.append(k - 1)
            prev = chrom[k]
    chr_ends.append(n_probes - 1)

    rows_chr = []
    rows_start = []
    rows_end = []
    rows_size = []
    rows_mean = []
    rows_psize = []

    for ordinal, (s, e) in enumerate(zip(chr_starts, chr_ends)):
        np_chr = e - s + 1
        if np_chr < 2:
            # 1-probe chromosome: upstream calc_std(...,1) divide-by-zero +
            # heap_max() NULL-deref crash (spec §1, vegaMC.c:159-161,451).
            raise ValueError(
                f"chromosome at appearance-ordinal {ordinal} has {np_chr} probe(s); "
                "vegaMC requires >=2 probes per chromosome (upstream crashes "
                "on 1-probe chromosomes)."
            )

        # data[sample, probe] for this chromosome (run_vegaMC.c:409-411)
        data_chr = np.ascontiguousarray(X[s : e + 1, :].T)  # (num_samples, np_chr)
        markers_start = np.arange(s, e + 1, dtype=np.int64)  # global probe idx

        # per-chromosome per-sample std, ddof=1 (run_vegaMC.c:417,676-684).
        # calc_std accumulates in float32; mimic with float32 throughout.
        std = _calc_std_per_sample(data_chr, np_chr)

        if not np.all(np.isfinite(std)):
            # 1-probe already guarded; this is the zero-elements / nan path.
            raise ValueError(
                f"chromosome at appearance-ordinal {ordinal}: non-finite std "
                "(empty or degenerate input)."
            )
        # NB: a zero-variance (flat) chromosome is NOT pre-empted here.  The C
        # kernel runs it fine (std==0 -> stop==0 -> no merges -> trivial
        # per-probe segments), which then drop out under the bp-size filter.
        # Verified against the R oracle: a flat chromosome alongside valid ones
        # yields segments only for the valid chromosomes, no error (fixture c9).
        # Whole-input zero variance (every segment filtered -> empty table) is
        # the genuine upstream-error case and is raised after the filter below
        # (fixture c7z).
        out_start, out_end, out_size, out_mean, n_reg = vega_mc_kernel(
            data_chr, markers_start, beta_eff, std, num_samples, weight, weight_sum
        )

        if n_reg == 0:
            raise ValueError(
                f"chromosome at appearance-ordinal {ordinal}: empty segmentation "
                "(zero-variance / upstream-error case, cf. c7z)."
            )

        for r in range(n_reg):
            start_idx = out_start[r]
            end_idx = out_end[r]
            bpst = int(pos[start_idx])
            bpe = int(pos[end_idx])
            bps = bpe - bpst + 1  # Size = End - Start + 1 (run_vegaMC.c:191)
            rows_chr.append(ordinal + 1)  # Chr = appearance-ordinal+1 (spec §1)
            rows_start.append(bpst)
            rows_end.append(bpe)
            rows_size.append(bps)
            rows_mean.append(float(out_mean[r]))
            rows_psize.append(int(out_size[r]))

    out = pd.DataFrame(
        {
            "Chr": rows_chr,
            "Start": rows_start,
            "End": rows_end,
            "Size": rows_size,
            "Mean": rows_mean,
            "Probe Size": rows_psize,
        }
    )

    # strict > filter on bp Size (run_vegaMC.c:204)
    out = out[out["Size"] > min_region_bp_size].reset_index(drop=True)

    if len(out) == 0:
        # Empty table after the strict bp filter: the R wrapper errors here
        # ("replacement has 1 row, data has 0") because the written file has no
        # data rows -- e.g. whole-input zero variance (fixture c7z). Raise
        # rather than fabricate. A single flat chromosome among valid ones is
        # NOT this case (its trivial segments drop, valid ones remain; c9).
        raise ValueError(
            "empty segmentation after min_region_bp_size filter "
            "(zero-variance / degenerate input; upstream SCEVAN errors here)."
        )

    return out


def _calc_std_per_sample(data_chr: np.ndarray, np_chr: int) -> np.ndarray:
    """Per-sample std over a chromosome, mimicking calc_std (run_vegaMC.c:676-684).

    calc_std = sqrt( sum (v - mean)^2 / (n-1) ), mean = sum(v)/n, float32
    accumulation; pow/sqrt in float64 then stored float32 (C casts to float).
    """
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

"""vegaMC orchestration layer (mirrors run_vegaMC.c call_VegaMC + driver).

Pure-Python driver around the Numba kernel in ``_vega_core``.  Splits the input
per chromosome (first-appearance order), computes per-chromosome per-sample std
(ddof=1), applies ``beta_eff = beta * num_samples`` once, runs the kernel per
chromosome, maps probe indices to genomic positions, and emits the segmentation
table.

T4 (here): per-segment loss/gain % counts + Loss/Gain Mean + focal scores
(DETERMINISTIC -> value-parity with R), and optional bootstrap p-values
(NON-parity: C uses libc rand(), uncorrelated across platforms -> we use
np.random.default_rng and document as consistency-parity only).  Also
``get_breaks_vegamc`` (vegaMC.R:119-127).

CRITICAL ordering (faithful to run_vegaMC.c): compute_matrices /
compute_pvalue run over ALL kernel segments BEFORE the strict
``Size > min_region_bp_size`` filter (write_segementation:204).  We therefore
build the full per-segment table first and apply the bp-size filter LAST.
"""

import numpy as np
import pandas as pd

from ._vega_core import _calc_mean_kernel, _calc_std_kernel, vega_mc_kernel


def vega_mc_r(
    mtx: pd.DataFrame,
    beta: float = 0.5,
    min_region_bp_size: int = 1000,
    loss_threshold: float = -0.2,
    gain_threshold: float = 0.2,
    with_pvalue: bool = False,
    bs: int = 1000,
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
        ``>`` kept, run_vegaMC.c:204).  Applied LAST, after the loss/gain %
        and p-values have been computed over all segments.
    loss_threshold, gain_threshold
        Per-sample segment-mean thresholds for the loss/gain % counts
        (run_vegaMC.c:342,347).  R defaults -0.2 / 0.2.
    with_pvalue
        If True, also compute bootstrap p-values (L.pv/G.pv).  These are
        NON-parity with R (libc rand() vs np.random) -- consistency only.
        If False, the p-value columns are emitted as NaN.
    bs
        Bootstrap iterations for the p-value null distribution (R default 1000).

    Returns
    -------
    DataFrame with columns
    ``["Chr","Start","End","Size","Mean","L.pv","G.pv","LOH.pv","X..L","X.G",
       "X.LOH","Probe Size","Loss Mean","Gain Mean","LOH Mean",
       "Focal-score Loss","Focal-score Gain","Focal-score LOH"]``
    matching the R seg_*.tsv layout.  The first 6 of the T3 columns
    (Chr/Start/End/Size/Mean/"Probe Size") are unchanged.
    """
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

    # Per-segment accumulators (over ALL kernel segments, pre bp-size filter).
    seg_chr = []
    seg_bpst = []
    seg_bpe = []
    seg_bps = []
    seg_mean = []
    seg_psize = []
    seg_gstart = []  # global probe index, segment start
    seg_gend = []  # global probe index, segment end

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
            start_idx = int(out_start[r])
            end_idx = int(out_end[r])
            bpst = int(pos[start_idx])
            bpe = int(pos[end_idx])
            bps = bpe - bpst + 1  # Size = End - Start + 1 (run_vegaMC.c:191)
            seg_chr.append(ordinal + 1)  # Chr = appearance-ordinal+1 (spec §1)
            seg_bpst.append(bpst)
            seg_bpe.append(bpe)
            seg_bps.append(bps)
            seg_mean.append(float(out_mean[r]))
            seg_psize.append(int(out_size[r]))
            seg_gstart.append(start_idx)
            seg_gend.append(end_idx)

    num_seg_regions = len(seg_chr)

    # ----------------------------------------------------------------------
    # compute_matrices (run_vegaMC.c:293-376): loss/gain counts + means,
    # over ALL segments, BEFORE the bp-size filter.  DETERMINISTIC.
    # ----------------------------------------------------------------------
    loss_thr = np.float32(loss_threshold)
    gain_thr = np.float32(gain_threshold)

    seg_loss_perc = np.zeros(num_seg_regions, dtype=np.int64)  # n_loss samples
    seg_gain_perc = np.zeros(num_seg_regions, dtype=np.int64)  # n_gain samples
    mean_loss = np.zeros(num_seg_regions, dtype=np.float64)
    mean_gain = np.zeros(num_seg_regions, dtype=np.float64)
    # per-sample aberration counts over all segments (for the null theta)
    num_loss_sample = np.zeros(num_samples, dtype=np.int64)
    num_gain_sample = np.zeros(num_samples, dtype=np.int64)

    for sidx in range(num_seg_regions):
        gs = seg_gstart[sidx]
        ge = seg_gend[sidx]
        size = ge - gs + 1
        n_loss = 0
        n_gain = 0
        acc_loss = np.float32(0.0)
        acc_gain = np.float32(0.0)
        for j in range(num_samples):
            # calc_mean over the segment's probes for sample j (float32 accum)
            m = _calc_mean(X[gs : ge + 1, j], size)
            if m < loss_thr:
                n_loss += 1
                num_loss_sample[j] += 1
                acc_loss = np.float32(acc_loss + m)
            if m > gain_thr:
                n_gain += 1
                num_gain_sample[j] += 1
                acc_gain = np.float32(acc_gain + m)
        seg_loss_perc[sidx] = n_loss
        seg_gain_perc[sidx] = n_gain
        mean_loss[sidx] = float(np.float32(acc_loss / n_loss)) if n_loss > 0 else 0.0
        mean_gain[sidx] = float(np.float32(acc_gain / n_gain)) if n_gain > 0 else 0.0

    # ----------------------------------------------------------------------
    # compute_pvalue (run_vegaMC.c:234-291): bootstrap null distribution.
    # NON-PARITY: C uses libc rand() (quantized /1000); we use np.random.
    # Documented as consistency-parity only.  Computed only on request.
    # ----------------------------------------------------------------------
    if with_pvalue:
        l_pv, g_pv = _bootstrap_pvalues(
            num_loss_sample,
            num_gain_sample,
            num_seg_regions,
            num_samples,
            seg_loss_perc,
            seg_gain_perc,
            bs,
        )
    else:
        l_pv = np.full(num_seg_regions, np.nan)
        g_pv = np.full(num_seg_regions, np.nan)

    # ----------------------------------------------------------------------
    # Formatting (vegaMC.R:63-90): percent strings, focal scores.
    # frac = seg_loss_perc / num_samples (write_segementation:194).
    # ----------------------------------------------------------------------
    frac_loss = seg_loss_perc.astype(np.float64) / num_samples
    frac_gain = seg_gain_perc.astype(np.float64) / num_samples

    # focal score (computed on the fraction, before the *100, vegaMC.R:63-74):
    #   f_l = frac_loss * abs(LossMean) / ProbeSize
    psize_arr = np.array(seg_psize, dtype=np.float64)
    focal_loss = frac_loss * np.abs(mean_loss) / psize_arr
    focal_gain = frac_gain * np.abs(mean_gain) / psize_arr

    pct_loss_str = [_pct_string(f) for f in frac_loss]
    pct_gain_str = [_pct_string(f) for f in frac_gain]
    pct_loh_str = ["0%"] * num_seg_regions  # LOH off in MVP

    out = pd.DataFrame(
        {
            "Chr": seg_chr,
            "Start": seg_bpst,
            "End": seg_bpe,
            "Size": seg_bps,
            "Mean": seg_mean,
            "L.pv": l_pv,
            "G.pv": g_pv,
            "LOH.pv": np.ones(num_seg_regions),  # LOH off (run_vegaMC.c:288)
            "X..L": pct_loss_str,
            "X.G": pct_gain_str,
            "X.LOH": pct_loh_str,
            "Probe Size": seg_psize,
            "Loss Mean": mean_loss,
            "Gain Mean": mean_gain,
            "LOH Mean": np.zeros(num_seg_regions),
            "Focal-score Loss": focal_loss,
            "Focal-score Gain": focal_gain,
            "Focal-score LOH": np.zeros(num_seg_regions),
        }
    )

    # strict > filter on bp Size (run_vegaMC.c:204), applied LAST.
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


def _bootstrap_pvalues(
    num_loss_sample: np.ndarray,
    num_gain_sample: np.ndarray,
    num_seg_regions: int,
    num_samples: int,
    seg_loss_perc: np.ndarray,
    seg_gain_perc: np.ndarray,
    bs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap p-values per compute_pvalue (run_vegaMC.c:234-291).

    NON-PARITY with R: the C code draws from libc ``rand()`` (quantized to
    /1000, run_vegaMC.c:688-693), which is platform/seed-dependent and cannot
    be reproduced cross-platform.  We use ``np.random.default_rng(0)`` Bernoulli
    draws instead, so these p-values are only CONSISTENCY-parity (same recipe,
    different RNG) -- they will NOT bit-match R.  Do not assert equality.

    theta_j = num_loss_sample[j] / num_seg_regions; for bs iterations sum a
    Bernoulli(theta_j) over samples -> null histogram of counts; cumulate from
    high; divide by bs; seg p = null[seg_loss_perc[s]].
    """
    rng = np.random.default_rng(0)
    theta_loss = num_loss_sample.astype(np.float64) / num_seg_regions
    theta_gain = num_gain_sample.astype(np.float64) / num_seg_regions

    # vectorised Bernoulli draws, quantized to /1000 with `<=` to mirror C
    # generate_binomial (run_vegaMC.c:688-693): r = floor(u*1000)/1000; r<=theta.
    # (Same recipe as C; the RNG stream still differs -> non-parity values.)
    u_loss = np.floor(rng.random((bs, num_samples)) * 1000.0) / 1000.0
    u_gain = np.floor(rng.random((bs, num_samples)) * 1000.0) / 1000.0
    loss_counts = (u_loss <= theta_loss).sum(axis=1)
    gain_counts = (u_gain <= theta_gain).sum(axis=1)

    null_loss = np.bincount(loss_counts, minlength=num_samples + 1).astype(np.float64)
    null_gain = np.bincount(gain_counts, minlength=num_samples + 1).astype(np.float64)

    # cumulate from high (run_vegaMC.c:269-273): null[j] += null[j+1]
    for j in range(num_samples - 1, -1, -1):
        null_loss[j] += null_loss[j + 1]
        null_gain[j] += null_gain[j + 1]
    null_loss /= bs
    null_gain /= bs

    # R rounds the p-value columns to 5 decimals (vegaMC.R:63-65).
    l_pv = np.round(null_loss[seg_loss_perc], 5)
    g_pv = np.round(null_gain[seg_gain_perc], 5)
    return l_pv, g_pv


def _pct_string(frac: float) -> str:
    """R-faithful percent string: paste0(round(frac*100, 1), "%").

    R's ``paste(as.numeric(round(x*100, 1)), "%")`` drops a trailing ``.0`` for
    whole numbers (60.0 -> "60%") but keeps one decimal otherwise (33.3 ->
    "33.3%").  We round to 1 decimal then format with %g-like trimming.
    """
    v = round(frac * 100.0, 1)
    # %g drops trailing zeros; for one-decimal values this yields R's format.
    if v == int(v):
        return f"{int(v)}%"
    return f"{v:g}%"


def _calc_mean(v: np.ndarray, n: int) -> np.float32:
    """calc_mean (run_vegaMC.c): sum(v)/n in float32 accumulation.

    Delegates to the njit ``_calc_mean_kernel`` (bit-exact float32 sequential
    sum; see tests/unit/test_vega_kernels.py). Re-wrapped in ``np.float32`` to
    restore the float32 scalar dtype numba widens on return.
    """
    return np.float32(_calc_mean_kernel(v, n))


def get_breaks_vegamc(
    mtx: pd.DataFrame,
    chr_vect: np.ndarray,
    beta_vega: float = 0.5,
) -> np.ndarray:
    """getBreaksVegaMC (vegaMC.R:119-127): segment, map Starts to indices.

    Runs ``vega_mc_r`` then locates each segment Start (a genomic position)
    inside ``chr_vect`` (the Position vector), returning the 0-based breakpoint
    indices.  R returns ``sort(unique(c(1, BR, n)))`` 1-based; we return the
    0-based equivalent ``sort(unique([0, BR0, n-1]))``.

    Parameters
    ----------
    mtx
        Same layout as ``vega_mc_r``.
    chr_vect
        The Position vector (R matches ``which(chr_vect == Start)``).
    beta_vega
        Segmentation beta (default 0.5).

    Returns
    -------
    np.ndarray of sorted unique 0-based breakpoint indices.
    """
    seg = vega_mc_r(mtx, beta=beta_vega)
    return breaks_from_starts(seg["Start"].to_numpy(), np.asarray(chr_vect), len(mtx))


def breaks_from_starts(
    starts: np.ndarray, chr_vect: np.ndarray, n: int
) -> np.ndarray:
    """Map segment Start positions to 0-based breakpoint indices.

    Extracted from ``get_breaks_vegamc`` so a caller that already has the seg
    table (e.g. classify_tumor_cells, which needs the p-value seg too) can
    derive breaks WITHOUT re-running ``vega_mc_r`` -- the deterministic ``Start``
    column is identical regardless of ``with_pvalue`` (the bootstrap touches
    only L.pv/G.pv). R: ``sort(unique(c(1, which(chr_vect==Start), n)))``.
    """
    br = []
    for start in starts:
        hits = np.where(chr_vect == start)[0]
        if hits.size:
            br.append(int(hits[0]))  # which(...)[1] -> first match, 0-based
    return np.array(sorted(set([0, *br, n - 1])))


def _calc_std_per_sample(data_chr: np.ndarray, np_chr: int) -> np.ndarray:
    """Per-sample std over a chromosome, mimicking calc_std (run_vegaMC.c:676-684).

    calc_std = sqrt( sum (v - mean)^2 / (n-1) ), mean = sum(v)/n, float32
    accumulation; pow/sqrt in float64 then stored float32 (C casts to float).

    Delegates to the njit ``_calc_std_kernel`` (bit-exact float32 sequential
    accumulation; see tests/unit/test_vega_kernels.py). ``data_chr`` is already
    C-contiguous at the call site (vegamc.py:113).
    """
    return _calc_std_kernel(data_chr, np_chr)

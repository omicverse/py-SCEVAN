"""EM 5-state copy-number calling.

Port of SCEVAN ``CNcall.R`` (getCNcall + the truncated-Gaussian EM machinery).
One R logical unit == one Python function; the line ranges below point back at
``SCEVAN-R/R/CNcall.R``.

The EM fits a 5-component truncated-Gaussian mixture (loss/sub-loss/neutral/
sub-gain/gain) to the per-segment mean relative expression, labels each segment
with the argmax-posterior state (0..4), then takes a per-chromosome majority CN
call.

Numerical contract: float64 throughout; ``scipy.stats.norm`` for dnorm/pnorm.

R ``max.col(posterior)`` uses ``ties.method="random"``. With continuous
posteriors exact ties are a measure-zero event, so we use ``np.argmax`` (first
maximum). The two agree except on that measure-zero set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["get_cn_call"]


# --------------------------------------------------------------------------- #
# getInitParameters  (CNcall.R:2-28)
# --------------------------------------------------------------------------- #
def _get_init_parameters(segm_vect, mean_vect, sd_vect, truncBoundRight, truncBoundLeft):  # noqa: N803
    mean_vect = np.asarray(mean_vect, dtype=np.float64).copy()
    sd_vect = np.asarray(sd_vect, dtype=np.float64).copy()
    segm_vect = np.asarray(segm_vect, dtype=np.float64)

    def bounds(mv, sv):
        left = np.array(
            [mv[1] * 3, mv[1] - 3 * sv[1], -truncBoundLeft, truncBoundRight, mv[3] + 3 * sv[3]],
            dtype=np.float64,
        )
        right = np.array(
            [mv[1] - 3 * sv[1], -truncBoundLeft, truncBoundRight, mv[3] + 3 * sv[3], mv[3] * 3],
            dtype=np.float64,
        )
        return left, right

    left, right = bounds(mean_vect, sd_vect)

    for i in range(len(mean_vect)):
        ind = np.where((segm_vect <= right[i]) & (segm_vect >= left[i]))[0]
        if len(ind) == 1:
            mean_vect[i] = segm_vect[ind[0]]
        elif len(ind) > 1:
            mean_vect[i] = np.mean(segm_vect[ind])
            # R sd() uses n-1 denominator.
            sd_vect[i] = np.std(segm_vect[ind], ddof=1)

    left, right = bounds(mean_vect, sd_vect)

    return {
        "meanVect": mean_vect,
        "sdVect": sd_vect,
        "truncBoundLeftVect": left,
        "truncBoundRightVect": right,
    }


# --------------------------------------------------------------------------- #
# getTruncatedGaussianDensity  (CNcall.R:68-71)  EQ16
# --------------------------------------------------------------------------- #
def _truncated_gaussian_density(x, mean, sd, truncBoundLeft, truncBoundRight):  # noqa: N803
    x = np.asarray(x, dtype=np.float64)
    num = norm.pdf(x, mean, sd) * (x <= truncBoundRight) * (x >= truncBoundLeft)
    deno = norm.cdf(truncBoundRight, mean, sd) - norm.cdf(truncBoundLeft, mean, sd)
    return num / deno


# --------------------------------------------------------------------------- #
# getConditionalProbabilities  (CNcall.R:74-77)  EQ18
# --------------------------------------------------------------------------- #
def _conditional_probabilities(prior, normaldata_mat):
    # prior[x] * normaldata_mat[:, x]  (broadcast over rows)
    return normaldata_mat * np.asarray(prior, dtype=np.float64)[np.newaxis, :]


# --------------------------------------------------------------------------- #
# getPosteriorProbability  (CNcall.R:80-88)
# --------------------------------------------------------------------------- #
def _posterior_probability(segm_vect, mean_vect, sd_vect, prior):
    segm_vect = np.asarray(segm_vect, dtype=np.float64)
    mean_vect = np.asarray(mean_vect, dtype=np.float64)
    sd_vect = np.asarray(sd_vect, dtype=np.float64)

    # normaldataMat[:, k] = dnorm(segmVect, meanVect[k], sdVect[k])
    normaldata_mat = norm.pdf(
        segm_vect[:, np.newaxis], mean_vect[np.newaxis, :], sd_vect[np.newaxis, :]
    )
    tau_mat = _conditional_probabilities(prior, normaldata_mat)
    posterior = tau_mat / tau_mat.sum(axis=1, keepdims=True)
    return posterior


# --------------------------------------------------------------------------- #
# EStep  (CNcall.R:91-128)
# --------------------------------------------------------------------------- #
def _e_step(segm_vect, mean_vect, sd_vect, prior, left_vect, right_vect):
    segm_vect = np.asarray(segm_vect, dtype=np.float64)
    n = len(segm_vect)
    k = len(mean_vect)

    cols = []
    for x in range(k):
        mu = mean_vect[x]
        sdev = sd_vect[x]
        lo = left_vect[x]
        up = right_vect[x]
        if np.sum((segm_vect <= up) & (segm_vect >= lo)) != 0:
            vec = _truncated_gaussian_density(segm_vect, mu, sdev, lo, up)
            vec = np.where(np.isinf(vec), 100.0, vec)
        else:
            vec = np.zeros(n, dtype=np.float64)
        cols.append(vec)
    normaldata_mat = np.column_stack(cols)

    tau_mat = _conditional_probabilities(prior, normaldata_mat)

    deno = tau_mat.sum(axis=1)
    ind0 = np.where(deno == 0)[0]
    if len(ind0) != 0:
        for ind in ind0:
            indmin = int(np.argmin(np.abs(np.asarray(mean_vect) - segm_vect[ind])))
            tau_mat[ind, indmin] = 1.0
            deno[ind] = 1.0

    taux = tau_mat / deno[:, np.newaxis]
    return taux


# --------------------------------------------------------------------------- #
# MStep  (CNcall.R:131-154)
# --------------------------------------------------------------------------- #
def _m_step(segm_vect, taux, mean_vect, sd_vect):
    segm_vect = np.asarray(segm_vect, dtype=np.float64)
    mean = np.asarray(mean_vect, dtype=np.float64).copy()
    sdev = np.asarray(sd_vect, dtype=np.float64).copy()
    k = taux.shape[1]

    prior_temp = taux.sum(axis=0)
    prior = np.full(k, 1e-06, dtype=np.float64)
    for j in range(k):
        if prior_temp[j] != 0:
            mean[j] = (taux[:, j] @ segm_vect) / prior_temp[j]  # EQ19
            sdev_new = np.sqrt((taux[:, j] @ ((segm_vect - mean[j]) ** 2)) / prior_temp[j])  # EQ20
            if sdev_new > 0:
                sdev[j] = sdev_new
            prior[j] = prior_temp[j] / len(segm_vect)  # EQ21

    prior = prior / prior.sum()
    return {"mu": mean, "sdev": sdev, "prior": prior}


# --------------------------------------------------------------------------- #
# EM  (CNcall.R:31-64)
# --------------------------------------------------------------------------- #
def _em(segm_vect, mean_vect, sd_vect, truncBoundRight, truncBoundLeft, prior):  # noqa: N803
    init = _get_init_parameters(segm_vect, mean_vect, sd_vect, truncBoundRight, truncBoundLeft)
    mean_vect = init["meanVect"]
    sd_vect = init["sdVect"]
    left_vect = init["truncBoundLeftVect"]
    right_vect = init["truncBoundRightVect"]
    prior = np.asarray(prior, dtype=np.float64).copy()

    likeli_new = float(np.sum(_posterior_probability(segm_vect, mean_vect, sd_vect, prior) * prior))

    threshold = 1e-03
    last_iter = 0
    for i in range(1, 101):
        last_iter = i
        likeli_old = likeli_new
        taux = _e_step(segm_vect, mean_vect, sd_vect, prior, left_vect, right_vect)
        mres = _m_step(segm_vect, taux, mean_vect, sd_vect)
        mean_vect = mres["mu"]
        sd_vect = mres["sdev"]
        prior = mres["prior"]

        likeli_new = float(
            np.sum(_posterior_probability(segm_vect, mean_vect, sd_vect, prior) * prior)
        )
        if abs(likeli_new - likeli_old) < threshold:
            break

    return {
        "meanVect": mean_vect,
        "sdVect": sd_vect,
        "prior": prior,
        "iter": last_iter,
    }


# --------------------------------------------------------------------------- #
# getExtractSegm  (CNcall.R:157-194)
# --------------------------------------------------------------------------- #
def _extract_segm(count_mtx, breaks):
    """Per cell, per segment, the mean of ``count_mtx`` over the segment rows.

    ``breaks`` are 0-based (sorted, including 0 and n-1). Mirrors R's 1-based
    ``startBreaks = breaks[:-1]`` / ``endBreaks = c(startBreaks[1:] - 1, n)``.

    Returns a DataFrame with columns cell, startseg, endseg, mean, sdVect where
    cell/startseg/endseg are 0-based indices (cell into columns, seg into rows).
    """
    mat = np.asarray(count_mtx, dtype=np.float64)
    n = mat.shape[0]
    breaks = np.asarray(breaks, dtype=np.int64)

    start_breaks = breaks[:-1]
    end_breaks = np.concatenate([start_breaks[1:] - 1, [n - 1]])

    n_cells = mat.shape[1]
    n_seg = len(start_breaks)

    rows = []
    for cell in range(n_cells):
        col = mat[:, cell]
        for i in range(n_seg):
            s = start_breaks[i]
            e = end_breaks[i]
            mean_value = col[s : e + 1].mean()
            rows.append((cell, int(s), int(e), mean_value, 0.0))

    return pd.DataFrame(rows, columns=["cell", "startseg", "endseg", "mean", "sdVect"])


# --------------------------------------------------------------------------- #
# getLabelCall  (CNcall.R:218-227)
# --------------------------------------------------------------------------- #
def _label_call(posterior):
    call_results = np.argmax(posterior, axis=1)  # max.col - 1  (0-based)
    prob_results = posterior.max(axis=1)
    return call_results, prob_results


# --------------------------------------------------------------------------- #
# getCallingCN  (CNcall.R:230-280)
# --------------------------------------------------------------------------- #
def _calling_cn(annot_matrix, extract_segm, truncBoundRight, truncBoundLeft, mean_vect, organism="human"):  # noqa: N803
    sd_vect = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    prior = np.array([0.05, 0.1, 0.7, 0.1, 0.05], dtype=np.float64)

    segm_means = extract_segm["mean"].to_numpy(dtype=np.float64)

    results_em = _em(segm_means, mean_vect, sd_vect, truncBoundRight, truncBoundLeft, prior)

    posterior = _posterior_probability(
        segm_means, results_em["meanVect"], results_em["sdVect"], results_em["prior"]
    )
    call_results, prob_results = _label_call(posterior)

    # df_call: Chromosome/Start from annot[startseg], End from annot[endseg].
    start_idx = extract_segm["startseg"].to_numpy(dtype=np.int64)
    end_idx = extract_segm["endseg"].to_numpy(dtype=np.int64)

    annot = np.asarray(annot_matrix.values)
    df_call = pd.DataFrame(
        {
            "Chromosome": annot[start_idx, 0],
            "Start": annot[start_idx, 1],
            "End": annot[end_idx, 2],
            "Mean": segm_means,
            "CN": call_results,
            "ProbCall": prob_results,
        }
    )
    # Chromosome/Start coerced to numeric for grouping & comparison parity with R.
    df_call["Chromosome"] = pd.to_numeric(df_call["Chromosome"])
    df_call["Start"] = pd.to_numeric(df_call["Start"])
    df_call["End"] = pd.to_numeric(df_call["End"])

    tot_chr = 22 if organism == "human" else 19

    def majority_call(ch):
        calls = []
        df_ch = df_call[df_call["Chromosome"] == ch]
        for pos_ini in pd.unique(df_ch["Start"]):
            cn_vals = df_ch[df_ch["Start"] == pos_ini]["CN"]
            # R: as.numeric(names(which.max(table(CN)))). table() sorts factor
            # levels ascending; which.max returns the first maximum.
            counts = {}
            for v in cn_vals:
                counts[v] = counts.get(v, 0) + 1
            best_val = None
            best_cnt = -1
            for v in sorted(counts.keys()):
                if counts[v] > best_cnt:
                    best_cnt = counts[v]
                    best_val = v
            calls.append(best_val)
        return calls

    call = []
    for ch in range(1, tot_chr + 1):
        call.extend(majority_call(ch))
    call = np.asarray(call)

    # CNV: rows up to the last chr==totChr segment whose End == max End on totChr.
    df_tot = df_call[df_call["Chromosome"] == tot_chr]
    max_end = df_tot["End"].max()
    match_idx = np.where(
        (df_call["Chromosome"].to_numpy() == tot_chr) & (df_call["End"].to_numpy() == max_end)
    )[0]
    last = int(match_idx[0])  # R which(...)[1]

    cnv = df_call.iloc[: last + 1, :][["Chromosome", "Start", "End"]].copy()
    cnv["CN"] = call
    cnv.columns = ["Chr", "Pos", "End", "CN"]
    cnv = cnv.reset_index(drop=True)
    return cnv


# --------------------------------------------------------------------------- #
# getCNcall  (CNcall.R:283-301)
# --------------------------------------------------------------------------- #
def get_cn_call(matrix_seg, count_mtx_annot, breaks, clonal=False, organism="human"):  # noqa: N803 — R getCNcall defaults CLONAL=FALSE
    """5-state EM copy-number call.

    Parameters
    ----------
    matrix_seg : DataFrame (genes x cells)
        Tumor-cell body of the CNAmat (smoothed/segmented relative expression).
    count_mtx_annot : DataFrame
        Gene annotation, first three columns = Chr, Start, End (R cols 1,2,3).
    breaks : array-like of int
        0-based segment break indices (sorted, including 0 and n-1).
    clonal : bool, default True
        CLONAL flag selecting the truncation bounds / mean vector.
    organism : {"human", "mouse"}, default "human"

    Returns
    -------
    DataFrame with columns Chr, Pos, End, CN.
    """
    if clonal:
        trunc_bound_left = 0.05
        trunc_bound_right = 0.05
        mean_vect = np.array(
            [-trunc_bound_left * 4, -trunc_bound_left * 2, 0, trunc_bound_right * 2, trunc_bound_right * 4],
            dtype=np.float64,
        )
    else:
        trunc_bound_left = 0.1
        trunc_bound_right = 0.1
        mean_vect = np.array(
            [-trunc_bound_left * 2, -trunc_bound_left * 1.5, 0, trunc_bound_right * 1.5, trunc_bound_right * 2],
            dtype=np.float64,
        )

    extract_segm = _extract_segm(matrix_seg, breaks)

    cnv = _calling_cn(
        count_mtx_annot.iloc[:, [0, 1, 2]],
        extract_segm,
        trunc_bound_right,
        trunc_bound_left,
        mean_vect,
        organism=organism,
    )
    return cnv

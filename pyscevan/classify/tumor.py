"""classify_tumor_cells: tumour / non-malignant classification.

Port of ``classifyTumorCells`` (classifyTumor.R:170-353), the **user-normal**
path (``norm_cell_names`` provided, ``FIXED_NORMAL_CELLS=FALSE``,
``SEGMENTATION_CLASS=TRUE``, ``SMOOTH=TRUE``).  Steps (numbered to match the R
source):

  1. set.seed(1) -- the R seed.  On this path there is **no RNG** (baseline is a
     median, smoothing/segmentation/clustering are deterministic; the bootstrap
     p-values inside vegaMC use their own internal RNG, not R's global seed), so
     the seed is documented but has no Python effect.
  2. baseline (178-190): per-gene MEDIAN over normal cells; relat = count - basel.
  3. smooth (194-234): tanh nonlinear smoother per cell.
  4. segmentation (238-254): vegaMC breaks + segmAlt; computeCNAmtx.
  5. column-center CNA_mtx (262); ward.D hclust on t(CNA_mtx); cut 2;
     classifyCluster -> non-malignant cluster (higher normal fraction).
  6. relativise CNA_mtx and count_mtx_relat against the non-malignant mean,
     column-center, scale by 1/(0.5*(max-min)) (302-312).
  7. re-hclust ward.D on CNA_mtx_relat; cut 2; classifyCluster -> cellType_pred.
  8. tum_cells = malignant; CNAmat = cbind(annot[gene_id,seqnames,end], relat).

Linkage note (the T9 parity risk): R uses ``hclust(parDist euclidean,
method="ward.D")``.  scipy ``linkage(method='ward')`` is **ward.D2** (it squares
the input distances via Lance-Williams on squared euclidean).  Because ward.D2
applied to ``sqrt(d)`` reproduces ward.D applied to ``d``
(``ward.D2(sqrt(d)) == ward.D(d)``), we feed ``sqrt(pdist)`` to scipy 'ward'.
This is validated against the MGH106 reference (see tests/test_classify_parity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ..cna import compute_cna_mtx
from ..segment.vegamc import breaks_from_starts, vega_mc_r
from .smooth import nonlinear_smooth

__all__ = ["classify_tumor_cells", "ClassifyResult", "ward_d_cutree2"]


@dataclass
class ClassifyResult:
    """Output container for :func:`classify_tumor_cells`.

    Mirrors the R ``ress`` list (``tum_cells``, ``CNAmat``, ``confidentNormal``).
    """

    tum_cells: list[str]
    CNAmat: pd.DataFrame  # noqa: N815 (mirror R name)
    confident_normal: list[str] = field(default_factory=list)


def ward_d_cutree2(mtx: pd.DataFrame) -> dict[str, int]:
    """ward.D hierarchical clustering on ``t(mtx)``, cut into 2 clusters.

    ``mtx`` is genes x cells; clustering is over **cells** (columns), so we
    transpose. Returns a dict cell-name -> cluster id in {1, 2}.

    R: ``hclust(parDist(t(mtx), method="euclidean"), method="ward.D")`` then
    ``cutree(hcc, 2)``.  scipy's 'ward' is ward.D2; ``ward.D(d) ==
    ward.D2(sqrt(d))``, so we feed ``sqrt`` of the condensed euclidean
    distances. ``fcluster(..., 2, 'maxclust')`` mirrors ``cutree(.,2)``.
    """
    cells = list(mtx.columns)
    x = mtx.to_numpy(dtype=np.float64).T  # (cells, genes)
    d = pdist(x, metric="euclidean")
    z = linkage(np.sqrt(d), method="ward")
    labels = fcluster(z, t=2, criterion="maxclust")
    return {c: int(lab) for c, lab in zip(cells, labels)}


def _classify_cluster(
    hcc2: dict[str, int], norm_cell_names: list[str]
) -> dict[str, str]:
    """classifyCluster (classifyTumor.R:135-147).

    The cluster (1 or 2) with the higher fraction of confident-normal cells is
    "non malignant"; the other is "malignant". On ties (``perc_norm`` equal),
    R's ``which(perc_norm==max)`` returns BOTH indices and the subsequent
    ``hcc2 == clust_norm`` (a vector) recycles -- but in practice with two
    distinct fractions there is a unique max. We take the first max index to
    mirror R's vector-as-scalar use in the dominant (non-tie) case.
    """
    norm_set = set(norm_cell_names)
    members = {1: [], 2: []}
    for cell, cl in hcc2.items():
        members[cl].append(cell)

    def frac(cl: int) -> float:
        m = members[cl]
        if len(m) == 0:
            return 0.0
        return len(norm_set.intersection(m)) / len(m)

    perc = [frac(1), frac(2)]
    # which(perc==max(perc))[1] -> 1-based; here clust 1 or 2
    clust_norm = 1 if perc[0] >= perc[1] else 2

    pred = {}
    for cell, cl in hcc2.items():
        pred[cell] = "non malignant" if cl == clust_norm else "malignant"
    return pred


def _column_center(mat: np.ndarray) -> np.ndarray:
    """apply(x, 2, function(x) x - mean(x)) -- subtract each column's mean."""
    return mat - mat.mean(axis=0, keepdims=True)


def classify_tumor_cells(
    count_mtx_norm: pd.DataFrame,
    count_mtx_annot: pd.DataFrame,
    norm_cell_names: list[str],
    sample: str = "",
    beta_vega: float = 0.5,
    fixed_normal_cells: bool = False,
    smooth: bool = True,  # noqa: N803
    segmentation_class: bool = True,  # noqa: N803
) -> ClassifyResult:
    """Classify tumour vs non-malignant cells (user-normal path).

    Parameters
    ----------
    count_mtx_norm
        genes x cells DataFrame of (normalised) counts. Gene index, cell
        columns.
    count_mtx_annot
        genes x 5 DataFrame with columns ``seqnames, start, end, gene_id,
        gene_name`` (same row order / count as ``count_mtx_norm``). R uses
        columns ``c(4, 1, 3)`` = ``gene_id, seqnames, end``.
    norm_cell_names
        Confident-normal cell names (subset of the columns).
    sample
        Sample label (unused on this path beyond the R seed).
    beta_vega
        vegaMC segmentation beta (default 0.5).
    fixed_normal_cells
        If True, the provided normals are forced to "non malignant" rather than
        clustering (MVP target is False).
    smooth
        Apply the tanh nonlinear smoother (default True, R ``SMOOTH``).
    segmentation_class
        Run vegaMC segmentation before clustering (default True, R
        ``SEGMENTATION_CLASS``).

    Returns
    -------
    ClassifyResult
        ``tum_cells`` (malignant cell names), ``CNAmat`` (genes x (3+cells):
        ``gene_id, seqnames, end`` + relativised body), ``confident_normal``.
    """
    # 1) R: set.seed(1). No RNG on this path -> documented, no Python effect.
    np.random.seed(1)

    if len(norm_cell_names) < 1:
        raise NotImplementedError(
            "synthetic-baseline path (no normal cells) not ported; this is the "
            "user-normal MVP. Provide norm_cell_names."
        )

    genes = count_mtx_norm.index
    cells = list(count_mtx_norm.columns)

    # 2) baseline = per-gene median over normal cells (apply(...,1,median)).
    present_norm = [c for c in norm_cell_names if c in count_mtx_norm.columns]
    if len(present_norm) == 1:
        basel = count_mtx_norm[present_norm[0]].to_numpy(dtype=np.float64)
    else:
        basel = count_mtx_norm[present_norm].median(axis=1).to_numpy(dtype=np.float64)

    relat_arr = count_mtx_norm.to_numpy(dtype=np.float64) - basel[:, None]

    # 3) smooth.
    if smooth:
        relat_arr = nonlinear_smooth(relat_arr)

    count_mtx_relat = pd.DataFrame(relat_arr, index=genes, columns=cells)

    # 4) segmentation.
    annot3 = count_mtx_annot[["gene_id", "seqnames", "end"]]
    segm = segmentation_class and len(norm_cell_names) > 0
    if segm:
        mtx_vega = pd.concat(
            [
                annot3.rename(
                    columns={"gene_id": "Name", "seqnames": "Chr", "end": "Position"}
                ).reset_index(drop=True),
                count_mtx_relat.reset_index(drop=True),
            ],
            axis=1,
        )
        chr_vect = count_mtx_annot["end"].to_numpy()
        # Single segmentation: vega_mc_r runs once (with_pvalue=True so segm_alt
        # below has its bootstrap terms); breaks derive from the SAME seg's
        # deterministic Start column. Avoids a redundant full vegaMC pass (the
        # old get_breaks_vegamc re-ran the identical segmentation). bit-exact:
        # Start is pre-bootstrap, so identical to a with_pvalue=False run.
        seg = vega_mc_r(mtx_vega, beta=beta_vega, with_pvalue=True)
        breaks = breaks_from_starts(
            seg["Start"].to_numpy(), np.asarray(chr_vect), len(mtx_vega)
        )
        # segmAlt = abs(Mean)>0.05 | (G.pv<0.01 | L.pv<0.01). The G.pv/L.pv terms
        # come from a NON-parity bootstrap (see vegamc._bootstrap_pvalues) and so
        # may diverge from R on borderline segments; abs(Mean)>0.05 is
        # deterministic and dominant.
        segm_alt = (
            (seg["Mean"].abs() > 0.05).to_numpy()
            | (seg["G.pv"] < 0.01).to_numpy()
            | (seg["L.pv"] < 0.01).to_numpy()
        )
        cna_arr = compute_cna_mtx(count_mtx_relat, breaks, segm_alt).to_numpy()
        cna_mtx = pd.DataFrame(cna_arr, index=genes, columns=cells)
    else:
        cna_mtx = count_mtx_relat.copy()

    # 5) column-center CNA_mtx; ward.D cut 2; classifyCluster.
    cna_centered = pd.DataFrame(
        _column_center(cna_mtx.to_numpy(dtype=np.float64)), index=genes, columns=cells
    )
    hcc2 = ward_d_cutree2(cna_centered)
    if fixed_normal_cells:
        norm_set = set(norm_cell_names)
        cell_type_pred = {
            c: "non malignant" if c in norm_set else "malignant" for c in cells
        }
    else:
        cell_type_pred = _classify_cluster(hcc2, norm_cell_names)

    # 6) relativise against the non-malignant mean.
    non_malig = [c for c in cells if cell_type_pred[c] == "non malignant"]
    cna_vals = cna_centered.to_numpy(dtype=np.float64)
    nm_mean = cna_vals[:, [cells.index(c) for c in non_malig]].mean(axis=1)
    cna_relat = cna_vals - nm_mean[:, None]
    cna_relat = _column_center(cna_relat)
    cna_relat = cna_relat / (0.5 * (cna_relat.max() - cna_relat.min()))
    cna_relat = pd.DataFrame(cna_relat, index=genes, columns=cells)

    if segm:
        cm = count_mtx_relat.to_numpy(dtype=np.float64)
        cm_nm = cm[:, [cells.index(c) for c in non_malig]].mean(axis=1)
        cm = cm - cm_nm[:, None]
        cm = _column_center(cm)
        cm = cm / (0.5 * (cm.max() - cm.min()))
        count_mtx_relat = pd.DataFrame(cm, index=genes, columns=cells)

    # 7) re-hclust ward.D on CNA_mtx_relat; cut 2; classifyCluster.
    hcc2b = ward_d_cutree2(cna_relat)
    if not fixed_normal_cells:
        cell_type_pred = _classify_cluster(hcc2b, norm_cell_names)

    tum_cells = [c for c in cells if cell_type_pred[c] == "malignant"]

    # 8) CNAmat = cbind(annot[gene_id,seqnames,end], body).
    if segm:
        body = count_mtx_relat
    else:
        body = cna_relat
    cnamat = pd.concat(
        [annot3.reset_index(drop=True), body.reset_index(drop=True)], axis=1
    )
    cnamat.index = genes

    return ClassifyResult(
        tum_cells=tum_cells,
        CNAmat=cnamat,
        confident_normal=list(norm_cell_names),
    )

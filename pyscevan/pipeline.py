"""pipeline_cna: top-level single-sample SCEVAN orchestration (MVP path).

Port of SCEVAN ``pipelineCNA`` (pipelineCNA.R:37-88) + ``getClonalCNProfile``
(pipelineCNA.R:91-125), restricted to the **MVP path**:

  * ``norm_cell`` provided (user-known confident normals; the auto-detect
    ``findConfident``/yaGST path is Phase 1.5),
  * ``SUBCLONES = False`` (subclone analysis is Phase 2),
  * no plotting, no ``segmTumorMatrix`` (cosmetic onlytumor segmentation), no
    file I/O.

Steps (numbered to mirror pipelineCNA.R):

  1. preprocess  -- preprocessing_mtx (pipelineCNA.R:45). R passes
     ``perc_genes/100``; SCEVANConfig.perc_genes is a *percentage* (10.0), so we
     divide by 100 here to hand preprocessing_mtx the 0.1 *fraction* it expects.
  2. classify    -- classify_tumor_cells (pipelineCNA.R:49), user-normal path
     (SEGMENTATION_CLASS=True, SMOOTH=True).
  3. classDf     -- per-cell class table (pipelineCNA.R:52-55): default
     "filtered" over the post-cell-filter count matrix, "normal" for every
     classified cell, "tumor" for tum_cells; a confidentNormal="yes" column for
     the user normals.
  4. clonal CN   -- getClonalCNProfile (pipelineCNA.R:91-116): re-segment the
     tumor-cell CNAmat with beta_vega=3, then EM CN-call (CLONAL=True).
  5. return      -- SCEVANResult(class_df, cna_matrix, clonal_cn).

Honesty notes
-------------
* The R ``getClonalCNProfile`` additionally ``cbind``s a cosmetic ``segm.mean``
  column read back from the beta=3 vega_output file (pipelineCNA.R:113-114).
  That column is **NOT** reproduced here -- our reference ``t7_clonalCN`` is the
  raw ``getCNcall`` output (Chr, Pos, End, CN), which is what we return.
* ``SUBCLONES=True`` is accepted but ignored with a warning (Phase 2).
"""

from __future__ import annotations

import warnings

import pandas as pd

from .classify.cncall import get_cn_call
from .classify.tumor import classify_tumor_cells
from .preprocess import preprocessing_mtx
from .result import SCEVANResult
from .segment.vegamc import get_breaks_vegamc

__all__ = ["pipeline_cna"]


def pipeline_cna(
    count_mtx: pd.DataFrame,
    norm_cell: list[str] | None = None,
    sample: str = "",
    beta_vega: float = 0.5,
    subclones: bool = False,
    clonal_cn: bool = True,
    fixed_normal_cells: bool = False,  # noqa: N803-style R fidelity
    organism: str = "human",
    ngenes_chr: int = 5,
    perc_genes: float = 10.0,
) -> SCEVANResult:
    """Run the single-sample SCEVAN CNA pipeline (MVP, user-normal path).

    Parameters
    ----------
    count_mtx
        Raw count matrix, genes x cells (gene symbols or Ensembl ids on the
        index, cell names on the columns).
    norm_cell
        Confident-normal cell names. **Required in the MVP** (the auto-detect
        path is Phase 1.5); passing ``None``/empty raises.
    sample
        Sample label (carried through; no file output in the MVP).
    beta_vega
        vegaMC segmentation beta for the classification step (default 0.5).
    subclones
        Phase-2 flag; if True it is ignored with a warning (MVP is single-clone).
    clonal_cn
        If True, infer the clonal CN profile from all tumour cells.
    fixed_normal_cells
        If True, the provided normals are forced non-malignant (skip clustering
        re-assignment); MVP default False.
    organism
        ``"human"`` (22 autosomes) or ``"mouse"`` (19).
    ngenes_chr
        Minimum genes-per-chromosome for the per-cell preprocessing filter.
    perc_genes
        Minimum percent of cells expressing a gene (a *percentage*, R default
        10). Converted to the 0.1 fraction preprocessing_mtx expects.

    Returns
    -------
    SCEVANResult
        ``class_df`` (per-cell class + confidentNormal), ``cna_matrix`` (the
        classify CNAmat: gene_id/seqnames/end + relativised body), ``clonal_cn``
        (Chr/Pos/End/CN, or None when ``clonal_cn=False``).
    """
    if subclones:
        warnings.warn(
            "subclones=True is ignored: the MVP pipeline_cna is single-clone "
            "(SUBCLONES path is Phase 2).",
            stacklevel=2,
        )

    if norm_cell is None or len(norm_cell) == 0:
        raise NotImplementedError(
            "pipeline_cna MVP requires user-provided norm_cell; the "
            "auto-detect (findConfident/yaGST) path is Phase 1.5."
        )

    # 1) preprocess (pipelineCNA.R:45). perc_genes/100 -> fraction semantics.
    res_proc = preprocessing_mtx(
        count_mtx,
        sample,
        ngenes_chr=ngenes_chr,
        perc_genes=perc_genes / 100.0,
        find_confident=False,
        organism=organism,
    )

    # 2) classify (pipelineCNA.R:49). User-normal path.
    res_class = classify_tumor_cells(
        res_proc.count_mtx_norm,
        res_proc.count_mtx_annot,
        norm_cell_names=norm_cell,
        sample=sample,
        beta_vega=beta_vega,
        fixed_normal_cells=fixed_normal_cells,
        smooth=True,
        segmentation_class=True,
    )

    # 3) classDf (pipelineCNA.R:52-55).
    proc_cells = list(res_proc.count_mtx_norm.columns)
    class_df = pd.DataFrame({"class": "filtered"}, index=proc_cells)
    # CNAmat columns minus the 3 annotation cols == every classified cell.
    annot_cols = ("gene_id", "seqnames", "end")
    classified_cells = [c for c in res_class.CNAmat.columns if c not in annot_cols]
    class_df.loc[classified_cells, "class"] = "normal"
    class_df.loc[res_class.tum_cells, "class"] = "tumor"
    # confidentNormal column: "yes" for user normals, NaN otherwise (R's NA).
    class_df["confidentNormal"] = pd.NA
    present_norms = [c for c in res_class.confident_normal if c in class_df.index]
    class_df.loc[present_norms, "confidentNormal"] = "yes"

    # 4) clonal CN (getClonalCNProfile 91-116). Skipped when clonal_cn=False.
    clonal_cn_df = None
    if clonal_cn:
        tum_mtx = res_class.CNAmat[res_class.tum_cells]
        annot3 = res_class.CNAmat[list(annot_cols)].rename(
            columns={"gene_id": "Name", "seqnames": "Chr", "end": "Position"}
        )
        mtx_vega = pd.concat(
            [annot3.reset_index(drop=True), tum_mtx.reset_index(drop=True)],
            axis=1,
        )
        chr_vect = res_class.CNAmat["end"].to_numpy()
        # R: getClonalCNProfile defaults beta_vega=3 for the clonal re-segment.
        breaks_tumor = get_breaks_vegamc(mtx_vega, chr_vect=chr_vect, beta_vega=3)
        clonal_cn_df = get_cn_call(
            tum_mtx,
            res_proc.count_mtx_annot,
            breaks_tumor,
            clonal=True,
            organism=organism,
        )

    # 5) return.
    return SCEVANResult(
        class_df=class_df,
        cna_matrix=res_class.CNAmat,
        clonal_cn=clonal_cn_df,
    )

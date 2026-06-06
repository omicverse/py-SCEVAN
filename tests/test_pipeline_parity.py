"""R-parity test for the end-to-end pipeline_cna vs SCEVAN pipelineCNA (MGH106).

MVP path: user-provided norm_cell, SUBCLONES=False, ClonalCN=True.

Parity tiers:
  * TIER-3 (identity): per-cell ``class`` (filtered/normal/tumor) and the
    ``confidentNormal`` column EXACT vs the R classDf; breaks_tumor (0-based)
    == t7_breaks_tumor - 1.
  * TIER-4 (value): clonal CN ``CN`` column EXACT vs t7_clonalCN; Chr/Pos/End
    identity.

The clonal CN omits R's cosmetic ``segm.mean`` column (pipelineCNA.R:113-114);
the reference t7_clonalCN is the raw getCNcall output (Chr/Pos/End/CN).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan import pipeline_cna
from pyscevan.classify.tumor import classify_tumor_cells
from pyscevan.preprocess import preprocessing_mtx
from pyscevan.segment.vegamc import get_breaks_vegamc

_REF = Path(__file__).parent / "r_ref" / "mgh106"


@pytest.mark.skipif(
    not (_REF / "t10_classDf.tsv").exists(), reason="MGH106 T10 fixtures absent"
)
def test_pipeline_end2end():
    count_mtx = pd.read_csv(
        _REF / "mgh106_subset.tsv.gz", sep="\t", index_col=0, compression="gzip"
    )
    norm_cell = [line.strip() for line in open(_REF / "norm_cell.txt") if line.strip()]

    res = pipeline_cna(
        count_mtx,
        norm_cell=norm_cell,
        sample="MGH106s",
        subclones=False,
        clonal_cn=True,
    )

    class_ref = pd.read_csv(_REF / "t10_classDf.tsv", sep="\t", index_col=0)
    clonal_ref = pd.read_csv(_REF / "t7_clonalCN.tsv", sep="\t")
    breaks_ref_1based = np.loadtxt(_REF / "t7_breaks_tumor.txt", dtype=np.int64)

    # --- TIER-3: classDf class identity (align by cell index) ---
    assert len(res.class_df) == len(class_ref), (
        f"classDf rows {len(res.class_df)} != ref {len(class_ref)}"
    )
    assert set(res.class_df.index) == set(class_ref.index), "cell sets differ"

    got_class = res.class_df["class"]
    ref_class = class_ref["class"]
    aligned = got_class.reindex(ref_class.index)
    mismatch = aligned[aligned != ref_class]
    assert mismatch.empty, (
        f"class mismatch on {len(mismatch)} cells: {list(mismatch.index[:10])}"
    )
    # spot-check the expected counts (136 tumor, 64 normal).
    counts = aligned.value_counts().to_dict()
    assert counts.get("tumor") == 136, counts
    assert counts.get("normal") == 64, counts

    # --- TIER-3: confidentNormal column identity ---
    got_cn = res.class_df["confidentNormal"].reindex(class_ref.index)
    ref_cn = class_ref["confidentNormal"]
    # R writes NA -> pandas reads as NaN; our column uses pd.NA. Compare the
    # "yes" set + the NA set.
    got_yes = set(got_cn[got_cn == "yes"].index)
    ref_yes = set(ref_cn[ref_cn == "yes"].index)
    assert got_yes == ref_yes, (
        f"confidentNormal yes mismatch: got {len(got_yes)}, ref {len(ref_yes)}"
    )
    # everything else must be NA on both sides.
    assert got_cn.isna().sum() == ref_cn.isna().sum()

    # --- TIER-3: breaks_tumor reproduction (beta_vega=3) ---
    # Re-derive the inputs to getClonalCNProfile to check breaks independently.
    res_proc = preprocessing_mtx(
        count_mtx, "MGH106s", ngenes_chr=5, perc_genes=0.1, organism="human"
    )
    res_class = classify_tumor_cells(
        res_proc.count_mtx_norm,
        res_proc.count_mtx_annot,
        norm_cell_names=norm_cell,
        beta_vega=0.5,
        fixed_normal_cells=False,
    )
    tum_mtx = res_class.CNAmat[res_class.tum_cells]
    annot3 = res_class.CNAmat[["gene_id", "seqnames", "end"]].rename(
        columns={"gene_id": "Name", "seqnames": "Chr", "end": "Position"}
    )
    mtx_vega = pd.concat(
        [annot3.reset_index(drop=True), tum_mtx.reset_index(drop=True)], axis=1
    )
    breaks_tumor = get_breaks_vegamc(
        mtx_vega, chr_vect=res_class.CNAmat["end"].to_numpy(), beta_vega=3
    )
    np.testing.assert_array_equal(breaks_tumor, breaks_ref_1based - 1)

    # --- TIER-4: clonal CN identity ---
    assert res.clonal_cn is not None
    assert len(res.clonal_cn) == len(clonal_ref), (
        f"clonalCN rows {len(res.clonal_cn)} != ref {len(clonal_ref)}"
    )
    np.testing.assert_array_equal(
        res.clonal_cn["Chr"].to_numpy(), clonal_ref["Chr"].to_numpy()
    )
    np.testing.assert_array_equal(
        res.clonal_cn["Pos"].to_numpy(), clonal_ref["Pos"].to_numpy()
    )
    np.testing.assert_array_equal(
        res.clonal_cn["End"].to_numpy(), clonal_ref["End"].to_numpy()
    )
    np.testing.assert_array_equal(
        res.clonal_cn["CN"].to_numpy(), clonal_ref["CN"].to_numpy()
    )

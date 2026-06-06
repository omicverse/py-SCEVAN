"""R-parity (identity) tests for annotate_genes vs SCEVAN::annotateGenes.

Reference (Track-B, MGH106 subset):
  - tests/r_ref/mgh106/mgh106_subset.tsv.gz : INPUT count_mtx (genes x cells).
  - tests/r_ref/mgh106/out_annot.tsv.gz     : R annotateGenes output
    (cols [seqnames,start,end,gene_id,gene_name, <cells>], no index col).

annotate_genes only reorders/subsets the input, so we assert exact gene-set +
order identity plus value identity on the annotation cols and a sample of cell
columns for the first/last few genes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.io.annotation import annotate_genes

_REF = Path(__file__).parent / "r_ref" / "mgh106"
_MTX = _REF / "mgh106_subset.tsv.gz"
_ANNOT = _REF / "out_annot.tsv.gz"

_ANNOT_COLS = ["seqnames", "start", "end", "gene_id", "gene_name"]


def _load():
    mtx = pd.read_csv(_MTX, sep="\t", index_col=0, compression="gzip")
    out_r = pd.read_csv(_ANNOT, sep="\t", compression="gzip")
    return mtx, out_r


def test_annotate_identity():
    if not _MTX.exists() or not _ANNOT.exists():
        pytest.skip("mgh106 fixtures missing")

    mtx, out_r = _load()
    out_py = annotate_genes(mtx, "human")

    # length parity (R gave 16570 genes for this subset)
    assert len(out_py) == len(out_r), f"len py={len(out_py)} r={len(out_r)}"
    assert len(out_py) == 16570, f"expected 16570 genes, got {len(out_py)}"

    # gene_name set AND order identity
    assert list(out_py["gene_name"]) == list(out_r["gene_name"]), "gene_name order"
    assert set(out_py["gene_name"]) == set(out_r["gene_name"]), "gene_name set"

    # seqnames all in 1..22
    assert out_py["seqnames"].isin(range(1, 23)).all(), "seqnames out of 1..22"

    # annotation-column value identity (full)
    for col in _ANNOT_COLS:
        np.testing.assert_array_equal(
            out_py[col].to_numpy(), out_r[col].to_numpy(), err_msg=f"annot col {col}"
        )

    # cell-column value identity for first & last few genes, sampled cells
    cell_cols = list(out_r.columns[5:])
    sample_cells = cell_cols[:3] + cell_cols[-3:]
    head_tail = list(range(5)) + list(range(len(out_r) - 5, len(out_r)))
    for col in sample_cells:
        np.testing.assert_allclose(
            out_py.iloc[head_tail][col].to_numpy(),
            out_r.iloc[head_tail][col].to_numpy(),
            atol=1e-6,
            err_msg=f"cell col {col}",
        )


def test_annotate_no_coordinate_sort():
    """annotate_genes preserves input row order (no coordinate sort here).

    MGH106 input is alphabetical (A1BG, A1BG-AS1, A1CF, ...); seqnames jump
    (19,19,10,...), proving rows are NOT sorted by genomic coordinate.
    """
    if not _MTX.exists():
        pytest.skip("mgh106 fixture missing")
    mtx, _ = _load()
    out_py = annotate_genes(mtx, "human")
    # first three gene_names follow input alphabetical order, not coord order
    assert list(out_py["gene_name"][:3]) == ["A1BG", "A1BG-AS1", "A1CF"]
    seq = out_py["seqnames"].to_numpy()
    assert not np.all(seq[:-1] <= seq[1:]), "rows unexpectedly sorted by seqnames"

"""R-parity test for preprocessing_mtx vs SCEVAN::preprocessingMtx (MGH106)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.preprocess import preprocessing_mtx

_REF = Path(__file__).parent / "r_ref" / "mgh106"


def _read(name, index_col=0):
    return pd.read_csv(
        _REF / name, sep="\t", index_col=index_col, compression="gzip"
    )


@pytest.mark.skipif(
    not (_REF / "mgh106_subset.tsv.gz").exists(), reason="MGH106 fixtures absent"
)
def test_preprocess_values():
    mtx = _read("mgh106_subset.tsv.gz")
    res = preprocessing_mtx(mtx)

    norm_ref = _read("preproc_norm.tsv.gz")
    annot_ref = pd.read_csv(
        _REF / "preproc_annot.tsv.gz", sep="\t", compression="gzip"
    )

    # gene set AND order
    assert list(res.count_mtx_norm.index) == list(norm_ref.index)
    assert list(res.count_mtx_annot["gene_name"]) == list(annot_ref["gene_name"])

    # cells (columns)
    assert list(res.count_mtx_norm.columns) == list(norm_ref.columns)

    # shape sanity (R: 10566 genes x 200 cells)
    assert res.count_mtx_norm.shape == norm_ref.shape

    # values
    assert np.allclose(
        res.count_mtx_norm.values, norm_ref.values, atol=1e-5
    ), f"max abs dev {np.abs(res.count_mtx_norm.values - norm_ref.values).max()}"

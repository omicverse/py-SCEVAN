"""R-parity unit test for compute_cna_mtx vs SCEVAN::computeCNAmtx (MGH106)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.cna import compute_cna_mtx

_REF = Path(__file__).parents[1] / "r_ref" / "mgh106"


def _read_gz(name):
    return pd.read_csv(_REF / name, sep="\t", index_col=0, compression="gzip")


@pytest.mark.skipif(
    not (_REF / "t8_in_mtx.tsv.gz").exists(), reason="MGH106 t8 fixtures absent"
)
def test_compute_cna_mtx():
    count_mtx = _read_gz("t8_in_mtx.tsv.gz")
    out_ref = _read_gz("t8_out_cna.tsv.gz")

    # R dumps 1-based breaks -> convert to 0-based.
    breaks = np.loadtxt(_REF / "t8_breaks.txt", dtype=int) - 1
    segm_alt = np.loadtxt(_REF / "t8_segmAlt.txt", dtype=int).astype(bool)

    out = compute_cna_mtx(count_mtx, breaks, segm_alt)

    assert out.shape == out_ref.shape
    assert list(out.index) == list(out_ref.index)
    assert list(out.columns) == list(out_ref.columns)
    np.testing.assert_allclose(out.values, out_ref.values, atol=1e-6)

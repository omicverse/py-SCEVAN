"""R-parity test for EM 5-state CN-call vs SCEVAN getCNcall (MGH106)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.classify.cncall import get_cn_call

_REF = Path(__file__).parent / "r_ref" / "mgh106"


@pytest.mark.skipif(
    not (_REF / "t7_clonalCN.tsv").exists(), reason="MGH106 T7 fixtures absent"
)
def test_em_cncall():
    matrix_seg = pd.read_csv(
        _REF / "t7_in_tum_mtx.tsv.gz", sep="\t", index_col=0, compression="gzip"
    )
    annot = pd.read_csv(
        _REF / "t7_annot.tsv.gz", sep="\t", compression="gzip"
    )
    # R 1-based break indices -> 0-based for the get_cn_call API.
    breaks_1based = np.loadtxt(_REF / "t7_breaks_tumor.txt", dtype=np.int64)
    breaks_0based = breaks_1based - 1
    assert len(breaks_0based) == 61

    cnv = get_cn_call(matrix_seg, annot, breaks_0based, clonal=True, organism="human")

    ref = pd.read_csv(_REF / "t7_clonalCN.tsv", sep="\t")

    # shape
    assert len(cnv) == len(ref), f"row count {len(cnv)} != {len(ref)}"

    # Chr / Pos / End identity
    np.testing.assert_array_equal(cnv["Chr"].to_numpy(), ref["Chr"].to_numpy())
    np.testing.assert_array_equal(cnv["Pos"].to_numpy(), ref["Pos"].to_numpy())
    np.testing.assert_array_equal(cnv["End"].to_numpy(), ref["End"].to_numpy())

    # CN labels: EXACT value parity target
    np.testing.assert_array_equal(
        cnv["CN"].to_numpy(), ref["CN"].to_numpy()
    )

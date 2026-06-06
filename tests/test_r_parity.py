"""R-parity tests: pyscevan.segment.vega_mc_r vs SCEVAN vegaMC R oracle.

Fixtures in tests/r_ref/vegamc/: in_<cid>.tsv (inputs) + seg_<cid>.tsv (R seg
output).  We assert the deterministic breaks columns; X..L/X.G/p-values are T4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.segment.vegamc import vega_mc_r

_REF = Path(__file__).parent / "r_ref" / "vegamc"
_CASES = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]


@pytest.mark.parametrize("cid", _CASES)
def test_vegamc_breaks(cid):
    in_path = _REF / f"in_{cid}.tsv"
    seg_path = _REF / f"seg_{cid}.tsv"
    if not in_path.exists() or not seg_path.exists():
        pytest.skip(f"fixture {cid} missing")

    mtx = pd.read_csv(in_path, sep="\t")
    seg_r = pd.read_csv(seg_path, sep="\t")

    seg_py = vega_mc_r(mtx, beta=0.5)

    assert len(seg_py) == len(seg_r), (
        f"{cid}: seg count py={len(seg_py)} vs r={len(seg_r)}"
    )

    np.testing.assert_array_equal(
        seg_py["Chr"].to_numpy(), seg_r["Chr"].to_numpy(), err_msg=f"{cid} Chr"
    )
    np.testing.assert_array_equal(
        seg_py["Start"].to_numpy(), seg_r["Start"].to_numpy(), err_msg=f"{cid} Start"
    )
    np.testing.assert_array_equal(
        seg_py["End"].to_numpy(), seg_r["End"].to_numpy(), err_msg=f"{cid} End"
    )
    np.testing.assert_array_equal(
        seg_py["Size"].to_numpy(), seg_r["Size"].to_numpy(), err_msg=f"{cid} Size"
    )
    np.testing.assert_array_equal(
        seg_py["Probe Size"].to_numpy(),
        seg_r["Probe Size"].to_numpy(),
        err_msg=f"{cid} Probe Size",
    )
    np.testing.assert_allclose(
        seg_py["Mean"].to_numpy(),
        seg_r["Mean"].to_numpy(),
        atol=1e-4,
        err_msg=f"{cid} Mean",
    )


def test_vegamc_zero_variance_guard():
    """c7z: all-zero (zero-variance) input -> R upstream errors; we raise."""
    in_path = _REF / "in_c7z.tsv"
    if not in_path.exists():
        pytest.skip("c7z fixture missing")
    mtx = pd.read_csv(in_path, sep="\t")
    with pytest.raises(ValueError):
        vega_mc_r(mtx, beta=0.5)

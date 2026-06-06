"""R-parity tests: pyscevan.segment.vega_mc_r vs SCEVAN vegaMC R oracle.

Fixtures in tests/r_ref/vegamc/: in_<cid>.tsv (inputs) + seg_<cid>.tsv (R seg
output).  We assert the deterministic breaks columns; X..L/X.G/p-values are T4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.segment.vegamc import get_breaks_vegamc, vega_mc_r

_REF = Path(__file__).parent / "r_ref" / "vegamc"
_CASES = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]


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
        atol=1e-5,
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


@pytest.mark.parametrize("cid", _CASES)
def test_vegamc_lossgain(cid):
    """T4-A: loss/gain % strings + Loss/Gain Mean value-parity vs R."""
    in_path = _REF / f"in_{cid}.tsv"
    seg_path = _REF / f"seg_{cid}.tsv"
    if not in_path.exists() or not seg_path.exists():
        pytest.skip(f"fixture {cid} missing")

    mtx = pd.read_csv(in_path, sep="\t")
    seg_r = pd.read_csv(seg_path, sep="\t")
    seg_py = vega_mc_r(mtx, beta=0.5)

    # percent strings: exact array equality
    np.testing.assert_array_equal(
        seg_py["X..L"].to_numpy(), seg_r["X..L"].astype(str).to_numpy(),
        err_msg=f"{cid} X..L",
    )
    np.testing.assert_array_equal(
        seg_py["X.G"].to_numpy(), seg_r["X.G"].astype(str).to_numpy(),
        err_msg=f"{cid} X.G",
    )
    # Loss/Gain Mean: numeric value-parity
    np.testing.assert_allclose(
        seg_py["Loss Mean"].to_numpy(), seg_r["Loss Mean"].to_numpy(),
        atol=1e-4, err_msg=f"{cid} Loss Mean",
    )
    np.testing.assert_allclose(
        seg_py["Gain Mean"].to_numpy(), seg_r["Gain Mean"].to_numpy(),
        atol=1e-4, err_msg=f"{cid} Gain Mean",
    )


@pytest.mark.parametrize("cid", ["c1", "c2"])
def test_vegamc_pvalue_soft(cid):
    """T4-B: p-values are NON-parity (libc rand) -> soft consistency checks only.

    We never assert equality on p-values; just that the columns exist, are in
    [0,1], and strong-signal (100%) segments fall below 0.05.
    """
    in_path = _REF / f"in_{cid}.tsv"
    seg_path = _REF / f"seg_{cid}.tsv"
    if not in_path.exists() or not seg_path.exists():
        pytest.skip(f"fixture {cid} missing")

    mtx = pd.read_csv(in_path, sep="\t")
    seg_py = vega_mc_r(mtx, beta=0.5, with_pvalue=True)

    assert "L.pv" in seg_py.columns
    assert "G.pv" in seg_py.columns
    lpv = seg_py["L.pv"].to_numpy()
    gpv = seg_py["G.pv"].to_numpy()
    assert np.all((lpv >= 0) & (lpv <= 1)), f"{cid} L.pv out of [0,1]"
    assert np.all((gpv >= 0) & (gpv <= 1)), f"{cid} G.pv out of [0,1]"

    loss100 = seg_py["X..L"].to_numpy() == "100%"
    gain100 = seg_py["X.G"].to_numpy() == "100%"
    assert np.all(lpv[loss100] < 0.05), f"{cid} 100% loss seg L.pv >= 0.05"
    assert np.all(gpv[gain100] < 0.05), f"{cid} 100% gain seg G.pv >= 0.05"


def test_get_breaks_c1():
    """T4-C: getBreaksVegaMC golden check (R 1-based 1,80,131,200,251,300)."""
    in_path = _REF / "in_c1.tsv"
    if not in_path.exists():
        pytest.skip("c1 fixture missing")
    mtx = pd.read_csv(in_path, sep="\t")
    position = mtx["Position"].to_numpy()
    br = get_breaks_vegamc(mtx, position)
    np.testing.assert_array_equal(br, np.array([0, 79, 130, 199, 250, 299]))

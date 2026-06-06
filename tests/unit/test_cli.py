"""Unit smoke for the run-h5ad CLI + AnnData I/O adapters (Task 11).

Fast (no real pipeline run): the round-trip + obs extraction are exact, and the
CLI command is exercised with ``pipeline_cna`` monkeypatched so we assert it is
invoked with the count matrix and normals extracted from the AnnData, and that
the three result tables are written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from pyscevan.io.anndata_io import adata_to_count_mtx, normal_cells_from_obs


def _make_adata(dense=True):
    # 3 cells x 4 genes, distinct values so transpose is checkable.
    x = np.array(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"is_normal": ["normal", "tumor", "normal"]},
        index=["cellA", "cellB", "cellC"],
    )
    var = pd.DataFrame(index=["GENE1", "GENE2", "GENE3", "GENE4"])
    xx = x if dense else sparse.csr_matrix(x)
    return AnnData(X=xx, obs=obs, var=var)


@pytest.mark.parametrize("dense", [True, False])
def test_adata_to_count_mtx_roundtrip(dense):
    adata = _make_adata(dense=dense)
    mtx = adata_to_count_mtx(adata)
    # genes x cells
    assert list(mtx.index) == ["GENE1", "GENE2", "GENE3", "GENE4"]
    assert list(mtx.columns) == ["cellA", "cellB", "cellC"]
    # cellB column = row 1 of X = [5,6,7,8]
    np.testing.assert_array_equal(
        mtx["cellB"].to_numpy(), np.array([5.0, 6.0, 7.0, 8.0])
    )
    # GENE3 row = col 2 of X across cells = [3,7,11]
    np.testing.assert_array_equal(
        mtx.loc["GENE3"].to_numpy(), np.array([3.0, 7.0, 11.0])
    )


def test_normal_cells_from_obs_value():
    adata = _make_adata()
    normals = normal_cells_from_obs(adata, "is_normal", normal_value="normal")
    assert normals == ["cellA", "cellC"]


def test_normal_cells_from_obs_default_label():
    adata = _make_adata()
    # No normal_value -> string "normal" (case-insensitive) counts as normal.
    normals = normal_cells_from_obs(adata, "is_normal")
    assert normals == ["cellA", "cellC"]


def test_normal_cells_from_obs_boolean():
    adata = _make_adata()
    adata.obs["flag"] = [True, False, True]
    normals = normal_cells_from_obs(adata, "flag")
    assert normals == ["cellA", "cellC"]


def test_normal_cells_from_obs_missing_key():
    adata = _make_adata()
    with pytest.raises(KeyError):
        normal_cells_from_obs(adata, "nope")


def test_cli_run_h5ad_invokes_pipeline(tmp_path, monkeypatch):
    """The CLI extracts count_mtx + normals and writes the result tables."""
    import pyscevan
    from pyscevan.result import SCEVANResult
    from typer.testing import CliRunner

    from pyscevan.cli import app

    adata = _make_adata()
    h5ad = tmp_path / "in.h5ad"
    adata.write_h5ad(h5ad)

    captured = {}

    def fake_pipeline_cna(count_mtx, **kwargs):
        captured["count_mtx"] = count_mtx
        captured["kwargs"] = kwargs
        class_df = pd.DataFrame(
            {"class": ["normal", "tumor", "normal"],
             "confidentNormal": ["yes", pd.NA, "yes"]},
            index=["cellA", "cellB", "cellC"],
        )
        clonal = pd.DataFrame({"Chr": [1], "Pos": [100], "End": [200], "CN": [2]})
        cnamat = pd.DataFrame(
            {"gene_id": ["GENE1"], "seqnames": [1], "end": [10],
             "cellB": [0.5]}
        )
        return SCEVANResult(class_df=class_df, cna_matrix=cnamat, clonal_cn=clonal)

    # patch the name as imported inside the command body
    monkeypatch.setattr(pyscevan, "pipeline_cna", fake_pipeline_cna)

    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-h5ad",
            "--input", str(h5ad),
            "--normal-key", "is_normal",
            "--normal-value", "normal",
            "--output", str(out),
            "--sample", "smpl",
        ],
    )

    assert result.exit_code == 0, result.output

    # pipeline got the genes x cells matrix + extracted normals
    cm = captured["count_mtx"]
    assert list(cm.index) == ["GENE1", "GENE2", "GENE3", "GENE4"]
    assert list(cm.columns) == ["cellA", "cellB", "cellC"]
    assert captured["kwargs"]["norm_cell"] == ["cellA", "cellC"]
    assert captured["kwargs"]["sample"] == "smpl"
    assert captured["kwargs"]["subclones"] is False
    assert captured["kwargs"]["clonal_cn"] is True

    # three tables written
    assert (out / "smpl_classDf.tsv").exists()
    assert (out / "smpl_clonalCN.tsv").exists()
    assert (out / "smpl_CNAmat.tsv.gz").exists()

    # summary line mentions tumor/normal counts
    assert "1 tumor" in result.output
    assert "2 normal" in result.output

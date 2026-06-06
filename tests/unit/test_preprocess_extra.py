"""Extra coverage for annotate/preprocess edge paths flagged by the T5/T6
codex review: the (production-rare) gene_id branch of annotate_genes, and the
find_confident=True rejection in preprocessing_mtx (MVP user-normal path)."""

import numpy as np
import pandas as pd
import pytest

from pyscevan.io.annotation import annotate_genes
from pyscevan.io.sysdata import load_annotation
from pyscevan.preprocess import preprocessing_mtx


def test_annotate_gene_id_branch_selected_and_aligned():
    """An Ensembl-ID-indexed matrix takes the gene_id branch and stays aligned.

    (Symbol matrices take the gene_name branch, covered by the MGH106 fixture;
    this exercises the otherwise-untested gene_id path.)
    """
    edb = load_annotation("human")
    edb = edb[edb["seqnames"].astype(str).isin([str(i) for i in range(1, 23)])]
    ids = edb["gene_id"].drop_duplicates().tolist()[:50]
    cm = pd.DataFrame(
        np.arange(len(ids) * 2).reshape(-1, 2),
        index=ids,
        columns=["cellA", "cellB"],
    )
    out = annotate_genes(cm, "human")

    # gene_id branch was taken: all output gene_ids come from the input ids.
    assert set(out["gene_id"]).issubset(set(ids))
    assert out["seqnames"].isin(range(1, 23)).all()
    # per-gene counts stay aligned to their gene_id after the edb reorder.
    for gid in out["gene_id"].head(8):
        np.testing.assert_array_equal(
            out.loc[out["gene_id"] == gid, ["cellA", "cellB"]].to_numpy().ravel(),
            cm.loc[gid].to_numpy(),
        )


def test_preprocess_rejects_find_confident_true():
    """MVP only supports user-provided normals; find_confident=True must raise."""
    cm = pd.DataFrame(
        np.ones((10, 5)),
        index=[f"g{i}" for i in range(10)],
        columns=[f"c{j}" for j in range(5)],
    )
    with pytest.raises(NotImplementedError):
        preprocessing_mtx(cm, find_confident=True)

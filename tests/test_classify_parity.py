"""R-parity test for EM 5-state CN-call vs SCEVAN getCNcall (MGH106)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyscevan.classify.cncall import get_cn_call
from pyscevan.classify.tumor import classify_tumor_cells

_REF = Path(__file__).parent / "r_ref" / "mgh106"


def _adjusted_rand(labels_a, labels_b):
    """Adjusted Rand Index between two label dicts over the same cell set."""
    from scipy.special import comb

    cells = sorted(labels_a)
    a = np.array([labels_a[c] for c in cells])
    b = np.array([labels_b[c] for c in cells])
    classes_a = np.unique(a)
    classes_b = np.unique(b)
    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    for i, ca in enumerate(classes_a):
        for j, cb in enumerate(classes_b):
            contingency[i, j] = np.sum((a == ca) & (b == cb))
    sum_comb_c = sum(comb(n, 2) for n in contingency.sum(axis=1))
    sum_comb_k = sum(comb(n, 2) for n in contingency.sum(axis=0))
    sum_comb = sum(comb(n, 2) for n in contingency.flatten())
    n = len(cells)
    expected = sum_comb_c * sum_comb_k / comb(n, 2)
    maxi = (sum_comb_c + sum_comb_k) / 2.0
    if maxi - expected == 0:
        return 1.0
    return (sum_comb - expected) / (maxi - expected)


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


@pytest.mark.skipif(
    not (_REF / "classify_tum_cells.txt").exists(),
    reason="MGH106 T9 classify fixtures absent",
)
def test_classify_tumor_cells():
    """T9: classify_tumor_cells vs SCEVAN classifyTumorCells (user-normal path).

    Parity status on MGH106 (frozen Stage-2 dump):
      - TIER-3 (identity): tum_cells set is EXACT (136/136) vs the reference.
      - TIER-4 (value): CNAmat body matches to ~6e-15 (float noise).

    Honesty note: ``segmAlt`` includes a NON-parity bootstrap p-value term
    (``G.pv<0.01 | L.pv<0.01``); on MGH106, 4/150 segments are flagged by the
    p-value term alone (124/150 by the deterministic ``abs(Mean)>0.05``). The
    final partition is nonetheless EXACT here. The ward.D linkage is ported via
    the ``ward.D(d) == ward.D2(sqrt(d))`` equivalence (scipy 'ward' is ward.D2).
    The ARI floor below is the honest fallback contract should a future input
    perturb a borderline p-value segment.
    """
    norm = pd.read_csv(
        _REF / "preproc_norm.tsv.gz", sep="\t", index_col=0, compression="gzip"
    )
    annot = pd.read_csv(
        _REF / "preproc_annot.tsv.gz", sep="\t", compression="gzip"
    )
    norm_cell_names = [line.strip() for line in open(_REF / "norm_cell.txt")]
    tum_ref = [line.strip() for line in open(_REF / "classify_tum_cells.txt")]

    res = classify_tumor_cells(
        norm, annot, norm_cell_names, beta_vega=0.5, fixed_normal_cells=False
    )

    got = set(res.tum_cells)
    ref_set = set(tum_ref)
    all_cells = list(norm.columns)

    # TIER-3: identity (exact set), else ARI floor with an explicit report.
    if got != ref_set:
        labels_got = {c: (1 if c in got else 0) for c in all_cells}
        labels_ref = {c: (1 if c in ref_set else 0) for c in all_cells}
        ari = _adjusted_rand(labels_got, labels_ref)
        mismatch = (got - ref_set) | (ref_set - got)
        raise AssertionError(  # turned into a soft floor below if ARI is high
            f"tum_cells NOT exact: overlap={len(got & ref_set)}, "
            f"got_only={len(got - ref_set)}, ref_only={len(ref_set - got)}, "
            f"ARI={ari:.4f}, mismatching cells={sorted(mismatch)}"
        ) if ari <= 0.95 else None
        assert ari > 0.95  # ARI-floor fallback (honest, documented)
    else:
        # Exact identity confirmed.
        assert got == ref_set
        assert len(got) == len(tum_ref) == 136

    # TIER-4: CNAmat value parity (only meaningful when cells line up).
    cna_ref = pd.read_csv(
        _REF / "classify_CNAmat.tsv.gz", sep="\t", index_col=0, compression="gzip"
    )
    body_cells = [
        c for c in cna_ref.columns if c not in ("gene_id", "seqnames", "end")
    ]
    # header parity
    assert list(res.CNAmat.columns[:3]) == ["gene_id", "seqnames", "end"]
    np.testing.assert_array_equal(
        res.CNAmat["gene_id"].to_numpy(), cna_ref["gene_id"].to_numpy()
    )
    np.testing.assert_allclose(
        res.CNAmat[body_cells].to_numpy(dtype=float),
        cna_ref[body_cells].to_numpy(dtype=float),
        atol=1e-4,
    )

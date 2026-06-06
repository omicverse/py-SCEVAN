"""Benchmark metrics for pyscevan vs R SCEVAN (py↔R consistency only).

This module is benchmark-only (lives in ``scripts/``, never imported by the
package). It quantifies how closely the Python reconstruction reproduces R
SCEVAN on the four dimensions Jason asked for:

  1. speed         -> ``speedup`` (R wall-clock / py wall-clock)
  2. CNA matrix    -> ``cna_spearman`` (Spearman rho, py CNAmat vs R CNAmat)
  3. coverage      -> static, see ``NAMESPACE_PARITY.md`` (not computed here)
  4. tumor calls   -> ``tumor_consistency`` (ARI / Jaccard / overlap)

No external truth label is used. The 3CA ``cell_type`` column (incl. its own
"Malignant" call) is itself an algorithm output, so it is NOT a reference; the
only target is R-vs-reconstruction agreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

_ANNOT_COLS = ("gene_id", "seqnames", "end")


def _body(cna: pd.DataFrame) -> pd.DataFrame:
    """Drop the 3 leading annotation columns, leaving genes x cells."""
    cells = [c for c in cna.columns if c not in _ANNOT_COLS]
    return cna[cells]


def cna_spearman(py_cna: pd.DataFrame, r_cna: pd.DataFrame) -> dict:
    """Spearman rho between py and R CNA matrices over shared (gene, cell).

    Returns the overall rho (flattened over the shared block) plus the
    distribution of per-cell rho (each cell = a vector over shared genes).
    """
    py_b = _body(py_cna)
    r_b = _body(r_cna)
    # align rows (genes) and columns (cells) on the shared set, R order.
    genes = [g for g in r_cna["gene_id"] if g in set(py_cna["gene_id"])]
    py_b = py_b.copy()
    r_b = r_b.copy()
    py_b.index = py_cna["gene_id"].to_numpy()
    r_b.index = r_cna["gene_id"].to_numpy()
    cells = [c for c in r_b.columns if c in set(py_b.columns)]
    py_m = py_b.loc[genes, cells].to_numpy(dtype=float)
    r_m = r_b.loc[genes, cells].to_numpy(dtype=float)

    overall = spearmanr(py_m.ravel(), r_m.ravel()).correlation
    per_cell = np.array(
        [spearmanr(py_m[:, j], r_m[:, j]).correlation for j in range(py_m.shape[1])]
    )
    return {
        "cna_spearman_overall": float(overall),
        "cna_spearman_cell_mean": float(np.nanmean(per_cell)),
        "cna_spearman_cell_median": float(np.nanmedian(per_cell)),
        "cna_spearman_cell_min": float(np.nanmin(per_cell)),
        "n_genes_shared": len(genes),
        "n_cells_shared": len(cells),
    }


def tumor_consistency(
    py_tum: list[str], r_tum: list[str], all_cells: list[str]
) -> dict:
    """ARI / Jaccard / overlap between py and R tumour-cell calls.

    ``all_cells`` is the universe of classified cells (the CNAmat body); the
    binary tumour/non-tumour labelling over it drives the ARI.
    """
    py_set, r_set = set(py_tum), set(r_tum)
    universe = list(all_cells)
    lab_py = np.array([1 if c in py_set else 0 for c in universe])
    lab_r = np.array([1 if c in r_set else 0 for c in universe])
    inter = len(py_set & r_set)
    union = len(py_set | r_set)
    return {
        "tumor_ari": float(adjusted_rand_score(lab_r, lab_py)),
        "tumor_jaccard": float(inter / union) if union else 1.0,
        "tumor_overlap": int(inter),
        "tumor_py_only": int(len(py_set - r_set)),
        "tumor_r_only": int(len(r_set - py_set)),
        "tumor_n_py": int(len(py_set)),
        "tumor_n_r": int(len(r_set)),
    }


def speedup(py_sec: float, r_sec: float) -> float:
    """R wall-clock / py wall-clock; NaN if either is missing/non-positive."""
    if not (np.isfinite(py_sec) and np.isfinite(r_sec)) or py_sec <= 0:
        return float("nan")
    return float(r_sec / py_sec)
